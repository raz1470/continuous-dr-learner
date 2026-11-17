"""Run the README's Python code blocks, in order, as a new user would after `pip install drcurves`.

Usage: python tools/ci/readme_quickstart.py README.md [n_blocks]

Runs the first `n_blocks` blocks (default 2: the quick start and "Many groups") in one namespace,
with a non-interactive matplotlib backend. Fails if any block raises.
"""
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")

text = open(sys.argv[1]).read()
n_blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 2
blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)[:n_blocks]
if len(blocks) < n_blocks:
    sys.exit(f"Expected {n_blocks} python blocks in {sys.argv[1]}, found {len(blocks)}")

namespace = {"__name__": "__main__"}
for i, code in enumerate(blocks, 1):
    start = time.time()
    exec(compile(code, f"README block {i}", "exec"), namespace)
    print(f"README block {i} ran in {time.time() - start:.0f}s")
