import json
from pathlib import Path

from code_repair_lm.config import load_config
from code_repair_lm.data import build_synthetic_dataset
from code_repair_lm.evaluation import evaluate_model
from code_repair_lm.model import CodeRepairLM
from code_repair_lm.tokenizer import CodeTokenizer


def main():
    cfg = load_config('configs/default.json')
    examples = build_synthetic_dataset()
    tokenizer = CodeTokenizer(vocab_size=cfg.vocab_size)
    corpus = [ex.buggy_code + '\n' + ex.fixed_code + '\n' + ex.error_message + '\n' + ex.unit_tests for ex in examples]
    tokenizer.build_vocab(corpus)
    model = CodeRepairLM(
        vocab_size=max(cfg.vocab_size, len(tokenizer)),
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        block_size=cfg.block_size,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    )
    metrics = evaluate_model(model, tokenizer, examples, cfg)
    print(json.dumps(metrics, indent=2))
    Path('artifacts').mkdir(exist_ok=True)
    with open('artifacts/evaluation_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
