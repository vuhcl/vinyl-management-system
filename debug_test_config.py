"""
Run with: .venv/bin/python debug_test_config.py
"""
import glob
import yaml

# Find the test config written by conftest.py
patterns = [
    "/tmp/pytest-*/grader_test*/artifacts*/test_grader.yaml",
    "/private/tmp/pytest-*/grader_test*/artifacts*/test_grader.yaml",
]

found = []
for p in patterns:
    found.extend(glob.glob(p))

if not found:
    print("No test config found — run pytest first to create it")
else:
    path = sorted(found)[-1]  # most recent
    print(f"Found: {path}")
    config = yaml.safe_load(open(path))
    abbrevs = config["preprocessing"]["abbreviation_map"]
    print("\nAbbreviation map order in test config:")
    for k, v in abbrevs.items():
        print(f"  {k!r:10} -> {v!r}")
    
    # Check if vg++ comes before vg+
    keys = list(abbrevs.keys())
    idx_vgpp = keys.index("vg++") if "vg++" in keys else -1
    idx_vgp  = keys.index("vg+")  if "vg+"  in keys else -1
    print(f"\nvg++ index: {idx_vgpp}")
    print(f"vg+  index: {idx_vgp}")
    if idx_vgpp < idx_vgp:
        print("ORDER CORRECT: vg++ before vg+")
    else:
        print("ORDER WRONG: vg+ before vg++ — this causes the bug")
