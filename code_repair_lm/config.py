import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Config:
    vocab_size: int = 2048
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 256
    ff_dim: int = 512
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    epochs: int = 12
    eval_every: int = 100
    checkpoint_dir: str = "checkpoints"
    seed: int = 1337
    max_new_tokens: int = 128
    max_repair_iters: int = 3
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def load_config(path: str = "configs/default.json") -> Config:
    p = Path(path)
    if not p.exists():
        return Config()
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return Config(**raw)
