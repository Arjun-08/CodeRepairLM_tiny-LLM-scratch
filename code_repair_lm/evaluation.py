import difflib
import math
import re
import time
from collections import Counter

import torch


def compute_exact_match(pred, target):
    return 1.0 if pred.strip() == target.strip() else 0.0


def compute_edit_distance(a: str, b: str):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def extract_code_block(text):
    match = re.search(r"```python\s*(.*?)```", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def syntax_valid(code):
    try:
        compile(code, '<repair>', 'exec')
        return True
    except Exception:
        return False


def executable_code(code):
    if not syntax_valid(code):
        return False
    ns = {}
    try:
        exec(code, ns, ns)
        return True
    except Exception:
        return False


def unit_test_pass_rate(code, tests):
    if not tests.strip():
        return 1.0
    try:
        ns = {}
        exec(code, ns, ns)
        exec(tests, ns, ns)
        return 1.0
    except Exception:
        return 0.0


def compute_token_accuracy(pred, target):
    if not target:
        return 0.0
    pred_tokens = list(pred)
    target_tokens = list(target)
    max_len = max(len(pred_tokens), len(target_tokens))
    if max_len == 0:
        return 0.0
    matches = sum(1 for p, t in zip(pred_tokens, target_tokens) if p == t)
    return matches / max_len


def evaluate_model(model, tokenizer, examples, cfg, device=None):
    if device is None:
        device = torch.device(cfg.device)
    model.eval()
    losses = []
    logits_list = []
    exact_match = 0.0
    token_accuracy = 0.0
    edit_distances = []
    syntax_valid_count = 0
    executable_count = 0
    pass_count = 0
    repair_success = 0
    total = max(1, len(examples))
    with torch.no_grad():
        for ex in examples:
            text = f"BUG:\n{ex.buggy_code}\nDESC:\n{ex.bug_description}\nERR:\n{ex.error_message}\nTESTS:\n{ex.unit_tests}\nFIX:\n{ex.fixed_code}\n"
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) > cfg.block_size:
                ids = ids[:cfg.block_size]
            input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            target_ids = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            logits = model(input_ids, max_seq_len=cfg.block_size)
            shift_logits = logits[:, :-1, :]
            shift_targets = target_ids[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, model.vocab_size),
                shift_targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='mean',
            )
            losses.append(loss.item())
            generated = model.generate(input_ids, max_new_tokens=max(8, len(ids) // 2))
            decoded = tokenizer.decode(generated[0].tolist())
            pred = extract_code_block(decoded)
            target = extract_code_block(ex.fixed_code)
            exact_match += compute_exact_match(pred, target)
            token_accuracy += compute_token_accuracy(pred, target)
            edit_distances.append(compute_edit_distance(pred, target))
            if syntax_valid(pred):
                syntax_valid_count += 1
            if executable_code(pred):
                executable_count += 1
            if unit_test_pass_rate(pred, ex.unit_tests):
                pass_count += 1
            if pred.strip() == target.strip():
                repair_success += 1
    val_loss = sum(losses) / len(losses)
    perplexity = math.exp(val_loss)
    metrics = {
        'validation_loss': float(val_loss),
        'perplexity': float(perplexity),
        'token_accuracy': float(token_accuracy / total),
        'exact_match_accuracy': float(exact_match / total),
        'mean_edit_distance': float(sum(edit_distances) / total),
        'syntax_validity_rate': float(syntax_valid_count / total),
        'executable_code_rate': float(executable_count / total),
        'unit_test_pass_rate': float(pass_count / total),
        'repair_success_rate': float(repair_success / total),
    }
    return metrics
