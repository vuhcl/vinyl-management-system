"""
Run with: .venv/bin/python debug_abbrev.py
"""
import re

abbr_pairs = [
    ('vg++', 'very good plus'),
    ('vg+',  'very good plus'),
    ('vg',   'very good'),
]

patterns = []
for abbr, expansion in abbr_pairs:
    escaped = re.escape(abbr.lower())
    if abbr.endswith('+'):
        pat = re.compile(r'(?<!\w)' + escaped + r'(?!\+)', re.IGNORECASE)
    else:
        pat = re.compile(r'(?<!\w)' + escaped + r'(?!\w)', re.IGNORECASE)
    patterns.append((pat, expansion))
    print(f'abbr={abbr!r}  pattern={pat.pattern!r}')

print()

for test in ['VG+ record', 'VG++ sleeve', 'VG++ sleeve, VG+ record']:
    text = test
    print(f'input:  {text!r}')
    for pat, exp in patterns:
        new_text = pat.sub(exp, text)
        if new_text != text:
            print(f'  matched pattern={pat.pattern!r}')
        text = new_text
    print(f'output: {text!r}')
    print()
