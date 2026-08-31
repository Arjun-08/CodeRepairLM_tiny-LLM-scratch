from code_repair_lm.config import load_config
from code_repair_lm.data import build_synthetic_dataset
from code_repair_lm.model import CodeRepairLM
from code_repair_lm.tokenizer import CodeTokenizer
from code_repair_lm.training import batchify

cfg = load_config('configs/default.json')
examples = build_synthetic_dataset()
tok = CodeTokenizer(vocab_size=cfg.vocab_size)
corpus = [ex.buggy_code + '\n' + ex.fixed_code + '\n' + ex.error_message + '\n' + ex.unit_tests for ex in examples]
tok.build_vocab(corpus)
print('tokenizer size', len(tok), 'vocab_size', tok.vocab_size, 'maxid', max(tok.token_to_id.values()))
model = CodeRepairLM(vocab_size=len(tok), d_model=cfg.d_model, n_head=cfg.n_head, n_layer=cfg.n_layer, block_size=cfg.block_size, ff_dim=cfg.ff_dim, dropout=cfg.dropout)
print('model vocab', model.vocab_size)
xt, yt = batchify(examples[:2], tok, __import__('torch').device('cpu'), cfg.block_size)
print('batch shapes', xt.shape, yt.shape)
print('max input id', int(xt.max()), 'min input id', int(xt.min()))
logits = model(xt)
print('logits shape', logits.shape)
