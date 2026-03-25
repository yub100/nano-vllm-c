from collections import deque

from config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager
# scheduler职责：
# 1.处理上一个step生成的token，依据块是否满来分配新物理kvcache block给逻辑块（将token加入seq中就已经是写入块了，
# 只不过这里写入的是逻辑块，在gpu前向计算时会依据这个toke所在逻辑块找到对应的kvcache物理块，然后计算这个token的kv写入cache
# 而其本身不需要存储在cache，推理时会直接拿临时变量存储q）
# 2.负责生成需要prefill或decode的seqs
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # The further to the left, the higher the priority.      
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
    
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def is_finished(self):
        return not (self.waiting or self.running)
    
    # When seq is seized, this function is called.
    # Using waiting.appendleft ensures that seq will be processed first when signal arrives.
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []

        # The number of seqence that scheduled_seqs get.
        num_seqs = 0
        num_batched_tokens = 0

        # prefill
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # one LLMEngine.step is one batch
            # radical judgment is: num_batched_tokens + len(seq) - seq.num_cached_tokens > max_num_batched_tokens
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break

            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True
        
        # decode
        # Do not need judge if num_batched_tokens > max_num_batched_tokens,
        # because num_batched_tokens <= len(running) <<< max_num_batched_tokens.
        while self.running and num_seqs < self.max_num_seqs:
            # when decode, one seq per batch 
            seq = self.running.popleft()

            # The reason why use 'while' is that although once self.preempt(self.running.pop())
            # can deallocate seq'data in blockmanage, can not free block itself if ref > 1.
            # Thus cycle until there a free block.
            # The function of 'while' is deal with the token generated in pre-step.
            while not self.block_manager.can_append(seq):
                # seize
                if self.running:
                    self.preempt(self.running.pop())
                # self-blocking
                else:
                    self.preempt(seq)
                    break
            # Executing when no 'break'.
            else:
                num_seqs += 1
                # Append new block or add the token generated in the pre-step with no new block.
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)

            # if token_id == eos or exceed max_tokens， seq reasoning complete
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
