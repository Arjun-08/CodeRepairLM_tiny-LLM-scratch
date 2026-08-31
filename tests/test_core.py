import pytest
import torch

from code_repair_lm.tokenizer import CodeTokenizer
from code_repair_lm.model import CodeRepairLM


def test_tokenizer_roundtrip_and_special_tokens():
    tok = CodeTokenizer(vocab_size=128)
    text = "def add(x, y):\n    return x + y\n"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert decoded.strip() == text.strip()
    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1


def test_model_forward_shape_and_output():
    tok = CodeTokenizer(vocab_size=128)
    model = CodeRepairLM(
        vocab_size=128,
        d_model=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        ff_dim=64,
    )
    seq = [tok.bos_token_id, 10, 11, 12, 13, 14, 15, 16]
    logits = model(torch.tensor([seq]), max_seq_len=16)
    assert logits.shape[-1] == 128
    assert logits.ndim == 3
    assert logits.shape[0] == 1


def test_decode_suppresses_unknown_artifacts():
    tok = CodeTokenizer(vocab_size=128)
    ids = [tok.bos_token_id, 9999, tok.token_to_id['def'], tok.token_to_id['return']]
    decoded = tok.decode(ids)
    assert '<UNK>' not in decoded
    assert 'def' in decoded or 'return' in decoded
