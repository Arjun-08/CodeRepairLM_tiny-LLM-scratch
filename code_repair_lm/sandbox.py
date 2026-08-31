import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def safe_exec(code: str, tests: str = ""):
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = {"passed": False, "stdout": "", "stderr": "", "exception": None, "exit_code": 0}
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            namespace = {}
            if code.strip():
                exec(code, namespace, namespace)
            if tests.strip():
                exec(tests, namespace, namespace)
            result["passed"] = True
    except Exception as exc:
        result["exception"] = repr(exc)
        result["exit_code"] = 1
    result["stdout"] = stdout.getvalue()
    result["stderr"] = stderr.getvalue()
    return result


def run_case_in_sandbox(code: str, tests: str = ""):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'solution.py'
        path.write_text(code, encoding='utf-8')
        test_path = Path(tmpdir) / 'tests.py'
        if tests.strip():
            test_path.write_text(tests, encoding='utf-8')
        cmd = [sys.executable, '-m', 'py_compile', str(path)]
        compile_proc = subprocess.run(cmd, capture_output=True, text=True)
        if compile_proc.returncode != 0:
            return {"passed": False, "stderr": compile_proc.stderr, "stdout": compile_proc.stdout, "exception": compile_proc.stderr, "exit_code": compile_proc.returncode}
        if tests.strip():
            cmd = [sys.executable, str(test_path)]
            test_proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
            return {"passed": test_proc.returncode == 0, "stderr": test_proc.stderr, "stdout": test_proc.stdout, "exception": test_proc.stderr, "exit_code": test_proc.returncode}
        return {"passed": True, "stderr": "", "stdout": "", "exception": None, "exit_code": 0}


def _sanitize_generated_code(text: str) -> str:
    cleaned = text.replace("<UNK>", "")
    cleaned = cleaned.replace("<PAD>", "")
    cleaned = cleaned.replace("<BOS>", "")
    cleaned = cleaned.replace("<EOS>", "")
    cleaned = cleaned.replace("<MASK>", "")
    cleaned = cleaned.replace("BUG:\n", "")
    cleaned = cleaned.replace("DESC:\n", "")
    cleaned = cleaned.replace("ERR:\n", "")
    cleaned = cleaned.replace("TESTS:\n", "")
    cleaned = cleaned.replace("FIX:\n", "")
    cleaned = cleaned.strip()
    if "```python" in cleaned:
        match = re.search(r"```python\s*(.*?)```", cleaned, re.S)
        if match:
            cleaned = match.group(1).strip()
    return cleaned


def repair_with_sandbox(model, tokenizer, buggy_code: str, bug_description: str, error_message: str, tests: str, cfg, max_iters=3):
    histories = []
    best_result = None
    best_code = buggy_code
    for iteration in range(1, max_iters + 1):
        prompt = f"BUG:\n{buggy_code}\nDESC:\n{bug_description}\nERR:\n{error_message}\nTESTS:\n{tests}\nFIX:\n"
        enc = tokenizer.encode(prompt, add_bos=True, add_eos=True)
        input_ids = torch.tensor([enc], dtype=torch.long)
        generated = model.generate(input_ids, max_new_tokens=cfg.max_new_tokens)
        candidate = _sanitize_generated_code(tokenizer.decode(generated[0].tolist()))
        code = candidate.split('FIX:\n', 1)[-1].strip() if 'FIX:' in candidate else candidate.strip()
        if not code:
            code = buggy_code
        result = run_case_in_sandbox(code, tests)
        histories.append({"iteration": iteration, "candidate": code, "result": result})
        if result["passed"]:
            best_result = result
            best_code = code
            break
        if best_result is None or result['exit_code'] < best_result['exit_code']:
            best_result = result
            best_code = code
    return {"best_code": best_code, "history": histories, "passed": bool(best_result and best_result.get('passed')), "final_result": best_result}
