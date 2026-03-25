from llm import LLM
from sampling_params import SamplingParams
import os
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/gz-data/model/qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.7, max_token=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize = False,
            add_generate_prompt = True
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"answer: {output}")

if __name__ == "__main__":
    main()
