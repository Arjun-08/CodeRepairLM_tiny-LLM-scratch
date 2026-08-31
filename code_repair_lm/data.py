from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RepairExample:
    buggy_code: str
    bug_description: str = ""
    error_message: str = ""
    unit_tests: str = ""
    fixed_code: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


DEFAULT_EXAMPLES = [
    RepairExample(
        buggy_code="def add(a, b):\n    return a + c\n",
        bug_description="Typo in variable name caused wrong output.",
        error_message="NameError: name 'c' is not defined",
        unit_tests="def test_add():\n    assert add(2, 3) == 5\n",
        fixed_code="def add(a, b):\n    return a + b\n",
    metadata={"category": "variable-name"},
    ),
    RepairExample(
        buggy_code="def is_even(n):\n    if n % 2 == 0:\n        return True\n    else:\n        return False\n",
        bug_description="Function should return False for odd inputs, but logic is fine.",
        error_message="",
        unit_tests="def test_is_even():\n    assert is_even(4) is True\n    assert is_even(5) is False\n",
        fixed_code="def is_even(n):\n    if n % 2 == 0:\n        return True\n    else:\n        return False\n",
        metadata={"category": "baseline"},
    ),
    RepairExample(
        buggy_code="def max_of_two(a, b):\n    if a > b:\n        return a\n    else:\n        return b\n",
        bug_description="No bug; baseline correct implementation.",
        error_message="",
        unit_tests="def test_max_of_two():\n    assert max_of_two(4, 7) == 7\n",
        fixed_code="def max_of_two(a, b):\n    if a > b:\n        return a\n    else:\n        return b\n",
        metadata={"category": "baseline"},
    ),
    RepairExample(
        buggy_code="def sum_list(values):\n    total = 0\n    for value in values:\n        total += value\n    return total\n",
        bug_description="Missing edge case for empty list should return 0.",
        error_message="",
        unit_tests="def test_sum_list():\n    assert sum_list([]) == 0\n    assert sum_list([1, 2, 3]) == 6\n",
        fixed_code="def sum_list(values):\n    total = 0\n    for value in values:\n        total += value\n    return total\n",
        metadata={"category": "empty-list"},
    ),
    RepairExample(
        buggy_code="def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
        bug_description="Works for positives; bug for negative input should raise ValueError.",
        error_message="ValueError: n must be non-negative",
        unit_tests="def test_fib():\n    assert fib(0) == 0\n    assert fib(1) == 1\n    assert fib(5) == 5\n",
        fixed_code="def fib(n):\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
        metadata={"category": "input-validation"},
    ),
]


def build_synthetic_dataset() -> List[RepairExample]:
    return [
        example for example in DEFAULT_EXAMPLES
    ]
