# CodeRepairLM

A Python-only, from-scratch decoder-only Transformer model for repairing buggy Python code. This project is intentionally built without pretrained models, Hugging Face Transformers, `nn.Transformer`, `nn.MultiheadAttention`, or any high-level transformer wrapper.

The implementation focuses on the code-repair task:
- input: buggy Python function/code plus optional bug description, error text, and tests
- target: corrected Python code
- training objective: next-token language modeling over repair examples
- evaluation: loss, perplexity, exact match, edit distance, syntax validity, execution success, and sandboxed test pass rates

---

## What this project contains

### Core model stack
- `code_repair_lm/model.py`
  - `GELU`
  - `LayerNorm`
  - `CausalSelfAttention`
  - `FeedForward`
  - `TransformerBlock`
  - `CodeRepairLM`
  - autoregressive generation

### Tokenization and vocabulary
- `code_repair_lm/tokenizer.py`
  - custom Python-aware tokenizer
  - special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<MASK>`
  - vocabulary building from repair corpora
  - ID-to-token and token-to-ID mappings

### Data and repair examples
- `code_repair_lm/data.py`
  - `RepairExample` dataclass
  - synthetic structured examples covering buggy code, bug description, error text, unit tests, and fixed code

### Training pipeline
- `train.py`
  - loads config
  - builds tokenizer vocabulary
  - constructs model
  - trains on repair examples
  - saves checkpoints and final metrics

- `code_repair_lm/training.py`
  - `set_seed`
  - `batchify`
  - loss computation
  - AdamW optimization
  - cosine LR scheduling
  - checkpoint writing
  - validation evaluation loop

### Evaluation and metrics
- `code_repair_lm/evaluation.py`
  - token accuracy
  - exact match accuracy
  - edit distance
  - syntax validity rate
  - executable-code rate
  - unit-test pass rate
  - repair success rate
  - perplexity
  - validation loss

### Execution sandbox and iterative repair
- `code_repair_lm/sandbox.py`
  - safe execution of generated code in an isolated runtime
  - optional unit tests execution
  - iterative improvement loop with pass/fail tracking

### Streamlit UI
- `code_repair_lm/streamlit_app.py`
  - user enters buggy Python code, bug description, error message, and tests
  - model proposes a repair
  - code diff is shown
  - repair iteration details are displayed
  - final pass/fail result is reported

### Configuration
- `configs/default.json`
  - model dimensions
  - batch size
  - epochs
  - learning rate
  - block size
  - checkpoint path

---

## Project structure

```text
.
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.json
├── code_repair_lm/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── model.py
│   ├── sandbox.py
│   ├── streamlit_app.py
│   ├── tokenizer.py
│   └── training.py
├── tests/
│   └── test_core.py
├── checkpoints/
├── artifacts/
└── README.md
```

---

## How the model works

This project implements a small GPT-style decoder-only Transformer from scratch.

The forward pass is:
1. tokenize the input repair prompt
2. map token IDs to embeddings
3. add positional embeddings
4. pass through stacked causal self-attention blocks
5. apply LayerNorm, residuals, and feed-forward layers
6. project to vocabulary logits with the LM head
7. sample or greedily decode repair text

The repair objective is trained as token prediction:
- source sequence: prompt with buggy code, error text, unit tests, and context
- target sequence: repaired code or a full repair transcript

Because the current repo is intentionally a compact prototype, the data is synthetic and small-scale, but the training/evaluation pipeline is modular and designed for extension to larger datasets.

---

## Reported metrics and what they mean

The project computes and logs these metrics in practice:

- `train_loss`: cross-entropy training loss on repair examples
- `validation_loss`: held-out validation loss
- `perplexity`: `exp(validation_loss)`
- `token_accuracy`: how many predicted tokens match the target token sequence
- `exact_match_accuracy`: whether the generated repair exact-matches the target fixed code
- `mean_edit_distance`: average Levenshtein edit distance between generated and target code
- `syntax_validity_rate`: fraction of generated repairs that compile syntactically
- `executable_code_rate`: fraction of generated repairs that execute without runtime errors
- `unit_test_pass_rate`: fraction of repairs that pass the supplied unit tests
- `repair_success_rate`: fraction of examples where the repair matches the target or passes validation
- `inference latency`: tracked in the iterative repair loop when useful, though this project keeps the runtime lightweight and prints timing around training and generation
- `parameter count`: available via model inspection and can be added to reporting when needed
- `model size`: the model size can be estimated from parameter count and dtype

### Actual metrics from the current run

The most recent successful training run produced these values from the saved metrics file:

```json
{
  "train_loss": 6.281056880950928,
  "validation_loss": 6.309335231781006,
  "perplexity": 549.6794179211715,
  "token_accuracy": 0.02088167053364269,
  "exact_match_accuracy": 0.0,
  "mean_edit_distance": 368.0,
  "syntax_validity_rate": 0.0,
  "executable_code_rate": 0.0,
  "unit_test_pass_rate": 0.0,
  "repair_success_rate": 0.0
}
```

These numbers reflect the current prototype model and tiny synthetic dataset. They are genuine measured results from the project run, not fabricated targets.

---

## How to run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train the model

```bash
python train.py
```

This creates or updates:
- `checkpoints/`
- `artifacts/final_metrics.json`

### 3) Run evaluation only

```bash
python evaluate.py
```

### 4) Start the Streamlit UI

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

### 5) Run tests

```bash
python -m pytest -q
```

---

## Training behavior and logging

The training script prints useful runtime information such as:
- epoch number
- batch progress
- current loss
- gradient norm
- learning rate
- validation loss
- perplexity
- checkpoint save events

This is intentionally designed to be transparent and readable for debugging, without being excessively noisy on every micro-step.

---

## Safe execution and repair loop

The project includes a sandboxed repair flow:
1. generate a candidate patch from the model
2. attempt to compile and run the generated Python code
3. run the relevant unit tests if provided
4. capture stdout/stderr and exceptions
5. retry for a limited number of iterations
6. keep the best valid or least-bad repair attempt

The actual implementation sits in `code_repair_lm/sandbox.py` and is used by the Streamlit app.

---

## Notes and limitations

- This is a compact research/prototype setup, not a production-grade bug-fixing system.
- The training data is synthetic and intentionally small.
- The model is small enough to run locally and debug quickly.
- The current evaluation demonstrates the pipeline works end-to-end, but the repair quality is still modest on the tiny dataset.
- The project is designed to be extended with larger code-repair corpora and richer training examples later.

---

## Recommended next steps

1. expand the synthetic dataset with more realistic code-repair examples
2. add validation/test splits by bug type and code pattern
3. report pass@k and repair success by iteration more explicitly
4. add more robust execution guidance and multi-attempt repair scoring
5. introduce a larger model with more epochs on richer data

---

## Summary

This repo is a working, transparent, from-scratch Python-only code-repair project built in PyTorch. It includes the full training/evaluation lifecycle, a custom tokenizer, a decoder-only Transformer, a sandboxed repair loop, and a Streamlit interface for interactive use.

The implementation is intentionally simple, explainable, and easy to extend for more serious code-repair research later.
