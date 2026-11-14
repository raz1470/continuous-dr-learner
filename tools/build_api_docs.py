"""Write docs/api.md from the docstrings of the public API (drcurves.__all__)."""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drcurves  # noqa: E402

SECTIONS = [("One call", ["fit_dose_response", "HierarchicalDoseResponse"]),
            ("Models and weights", ["default_model", "GPS", "DiscreteGPS", "BalancingWeights"]),
            ("Checks", ["coverage_check", "bootstrap_curves", "placebo_check"]),
            ("Building blocks", ["cross_fit_nuisances", "pseudo_outcomes", "Nuisances"])]


def sig(obj, name):
    try:
        return f"{name}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        return name


out = ["# API reference", "", "Generated from the docstrings by `tools/build_api_docs.py`. Everything listed here is",
       "the supported public API; other modules (`drcurves.riesz`, `drcurves.simulate`) may change before 1.0.", ""]
for title, names in SECTIONS:
    out += [f"## {title}", ""]
    for n in names:
        obj = getattr(drcurves, n)
        out += [f"### `{n}`", "", "```python", sig(obj, n), "```", "", "```text", inspect.getdoc(obj) or "", "```", ""]
        if inspect.isclass(obj):
            for m, fn in inspect.getmembers(obj, inspect.isfunction):
                if m.startswith("_") or fn.__qualname__.split(".")[0] != obj.__name__:
                    continue
                out += [f"#### `.{m}`", "", "```python", sig(fn, m), "```", "", "```text", inspect.getdoc(fn) or "", "```", ""]
(ROOT / "docs" / "api.md").write_text("\n".join(out))
print("wrote docs/api.md")
