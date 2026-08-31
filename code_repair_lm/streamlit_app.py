import difflib
import json
from pathlib import Path

import streamlit as st
import torch

from .config import load_config
from .data import build_synthetic_dataset
from .model import CodeRepairLM
from .sandbox import repair_with_sandbox
from .tokenizer import CodeTokenizer


@st.cache_resource
def load_model_and_tokenizer():
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
    return cfg, tokenizer, model


def render_diff(orig, repaired):
    diff = list(difflib.unified_diff(orig.splitlines(True), repaired.splitlines(True), lineterm=''))
    return '\n'.join(diff)


def main():
    st.set_page_config(page_title='CodeRepairLM', layout='wide')
    st.title('CodeRepairLM: Python Code Repair')
    cfg, tokenizer, model = load_model_and_tokenizer()
    bug_text = st.text_area('Buggy Python code', value='def add(a, b):\n    return a + c\n')
    desc = st.text_area('Bug description', value='Typo in variable name.')
    err = st.text_area('Error message', value='NameError: name \'c\' is not defined')
    tests = st.text_area('Unit tests', value='def test_add():\n    assert add(2, 3) == 5\n')
    if st.button('Repair code'):
        with st.spinner('Running CodeRepairLM and sandbox validation...'):
            result = repair_with_sandbox(model, tokenizer, bug_text, desc, err, tests, cfg, max_iters=cfg.max_repair_iters)
            code = result['best_code']
            st.subheader('Generated patch')
            st.code(code)
            st.subheader('Original vs repaired')
            st.code(render_diff(bug_text, code))
            st.subheader('Repair iteration history')
            for item in result['history']:
                st.write(f"Iteration {item['iteration']}: passed={item['result']['passed']} exit_code={item['result']['exit_code']}")
                if item['candidate']:
                    st.code(item['candidate'])
            outcome = 'PASS' if result['passed'] else 'FAIL'
            st.success(f'Final repair status: {outcome}')
            if result['final_result']:
                st.write(result['final_result'])
