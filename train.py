import json
import time
from pathlib import Path

from code_repair_lm.config import load_config
from code_repair_lm.data import build_synthetic_dataset
from code_repair_lm.model import CodeRepairLM
from code_repair_lm.tokenizer import CodeTokenizer
from code_repair_lm.training import train_and_evaluate


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
    metrics = train_and_evaluate(model, tokenizer, examples, cfg)
    out_dir = Path('artifacts')
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / 'final_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print('Training completed. Metrics saved to artifacts/final_metrics.json')


if __name__ == '__main__':
    start = time.time()
    print('Starting CodeRepairLM training...')
    main()
    elapsed = time.time() - start
    print(f'Completed in {elapsed:.2f}s')
