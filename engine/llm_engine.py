import atexit
# used for obtaining all fields information from class
from dataclasses import fields
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from time import perf_counter

from config import Config
from engine.model_runner import ModelRunner
from engine.scheduler import Scheduler
from sampling_params import SamplingParams
from engine.sequence import Sequence

class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        
        # obtain the fields in config from kwargs
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        
        # initialize Config
        config  = Config(model, **config_kwargs)

        # child process list
        self.ps = []

        self.events = []
        
        # factory function
        ctx = mp.get_context("spawn")
        
        # range 0 is main process, thus only need create (tensor_parallel_size - 1) child processes
        for i in range(1, config.tensor_parallel_size):
            # used to correspond
            event = ctx.Event()

            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)

        # use_fast=True specifies Tokenizer as the Rust version.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        # register function and call it when atexit
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_param: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_param)
        self.scheduler.add(seq)

    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self):
        # seqs.shape likes [seq_num, seq's token num]
        seqs, is_prefill = self.scheduler.schedule()

        # since one run() only generate one token, thus token_ids.len = seqs.len(0)
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    # only after dealing this batch prompts can generate next batch prompts,
    # schedule only applies on one generate()'s prompts,
    # processing complete means scheduler's queue become empty.
    def generate(
            self,
            prompts: list[str] | list[list[int]],
            sampling_params: SamplingParams | list[SamplingParams],
            use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="genrating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = sampling_params * len(prompts)
        
        # break the prompts into individual tasks and put them to scheduler
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            start = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - start)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - start)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}token/s",
                    "Decode": f"{int(decode_throughput)}token/s"
                })
            for seq_id, token_ids in output:
                # map
                outputs[seq_id] = token_ids
                if use_tqdm:
                    # +1
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs


