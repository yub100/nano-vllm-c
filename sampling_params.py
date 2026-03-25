from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0

    # the quantity per prompt can generate
    max_token: int = 64

    ignore_eos: bool = False