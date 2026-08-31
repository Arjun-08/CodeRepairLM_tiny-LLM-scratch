import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import RepairExample
from .evaluation import evaluate_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batchify(examples, tokenizer, device, block_size):
    texts = []
    for ex in examples:
        context = f"BUG:\n{ex.buggy_code}\nDESC:\n{ex.bug_description}\nERR:\n{ex.error_message}\nTESTS:\n{ex.unit_tests}\nFIX:\n{ex.fixed_code}\n"
        texts.append(context)
    encoded = [tokenizer.encode(text, add_bos=True, add_eos=True) for text in texts]
    max_len = min(max(len(x) for x in encoded), block_size)
    batched = []
    for ids in encoded:
        ids = ids[:max_len]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [tokenizer.pad_token_id] * pad_len
        batched.append(ids)
    tensor = torch.tensor(batched, dtype=torch.long, device=device)
    x = tensor[:, :-1]
    y = tensor[:, 1:]
    return x, y


def compute_loss(logits, targets, ignore_index):
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.contiguous().view(-1)
    loss = nn.functional.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction='mean')
    return loss


def train_and_evaluate(model, tokenizer, examples, cfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs * max(1, len(examples) // max(1, cfg.batch_size))))
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_examples = examples[: max(1, len(examples) * 80 // 100)]
    val_examples = examples[max(1, len(examples) * 80 // 100):]
    if not val_examples:
        val_examples = train_examples[:1]
    metrics_history = []
    best_val = float('inf')
    best_state = None
    n_batches = max(1, len(train_examples) // max(1, cfg.batch_size))
    for epoch in range(cfg.epochs):
        random.shuffle(train_examples)
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for step in range(0, len(train_examples), cfg.batch_size):
            batch = train_examples[step: step + cfg.batch_size]
            x, y = batchify(batch, tokenizer, device, cfg.block_size)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, max_seq_len=cfg.block_size)
            loss = compute_loss(logits, y, ignore_index=tokenizer.pad_token_id)
            loss.backward()
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * x.size(0)
            if step % (max(1, cfg.eval_every * 2)) == 0:
                print(f"Epoch {epoch+1}/{cfg.epochs} | batch {step} | loss={loss.item():.4f} | grad_norm={grad_norm:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")
        avg_epoch_loss = epoch_loss / max(1, len(train_examples))
        val_metrics = evaluate_model(model, tokenizer, val_examples, cfg)
        metrics_history.append({"epoch": epoch + 1, "train_loss": avg_epoch_loss, "val_loss": val_metrics.get('validation_loss'), 'perplexity': val_metrics.get('perplexity')})
        print(f"Epoch {epoch+1}/{cfg.epochs} done | avg_train_loss={avg_epoch_loss:.4f} | val_loss={val_metrics.get('validation_loss'):.4f} | ppl={val_metrics.get('perplexity'):.4f} | time={time.time()-start:.2f}s")
        if val_metrics.get('validation_loss', float('inf')) < best_val:
            best_val = val_metrics.get('validation_loss', float('inf'))
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_dir / 'best_model.pt')
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt')
    final_val = evaluate_model(model, tokenizer, val_examples, cfg)
    results = {
        'train_loss': metrics_history[-1]['train_loss'],
        'validation_loss': final_val.get('validation_loss'),
        'perplexity': final_val.get('perplexity'),
        'token_accuracy': final_val.get('token_accuracy'),
        'exact_match_accuracy': final_val.get('exact_match_accuracy'),
        'eval_metrics': final_val,
        'history': metrics_history,
    }
    return results
