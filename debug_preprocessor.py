"""
Run with: .venv/bin/python debug_preprocessor.py
"""
import sys
sys.path.insert(0, '.')

from grader.src.data.preprocess import Preprocessor

p = Preprocessor(
    'grader/configs/grader.yaml',
    'grader/configs/grading_guidelines.yaml',
)

print("=== abbreviation_pairs (first 5) ===")
for pair in p.abbreviation_pairs[:5]:
    print(f"  {pair}")

print(f"\n=== n_patterns: {len(p.abbreviation_patterns)} ===")
for pat, exp in p.abbreviation_patterns[:5]:
    print(f"  pattern={pat.pattern!r}  expansion={exp!r}")

print("\n=== expansion tests ===")
tests = [
    "VG+ record",
    "VG++ sleeve",
    "VG++ sleeve, VG+ record",
    "NM sleeve",
    "vg+ media",
]
for text in tests:
    result = p.clean_text(text)
    status = "✓" if "very good plus" in result or "near mint" in result else "✗"
    print(f"  {status}  {text!r:35} -> {result!r}")
