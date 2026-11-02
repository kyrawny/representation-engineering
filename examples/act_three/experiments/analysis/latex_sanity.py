"""
Static LaTeX sanity checks (no TeX engine is available here; the user compiles the paper):
balanced braces, matching begin/end, every \\ref has a \\label, every citation key is in references.bib,
and every table row (a line with '&') in paper/generated/tab_*.tex and the main file ends with \\\\.

    python -m examples.act_three.experiments.analysis.latex_sanity
"""
import re
import sys
from collections import Counter

from .common import GENERATED_DIR, PAPER_DIR

MAIN = PAPER_DIR / "AnonymousSubmission2027.tex"


def strip_comments(s: str) -> str:
    return re.sub(r"(?m)(?<!\\)%.*$", "", s)


def main() -> int:
    bad = 0
    s = strip_comments(MAIN.read_text(encoding="utf-8"))
    print("brace balance (main):", s.count("{") - s.count("}"))
    b = Counter(re.findall(r"\\begin\{(\w+\*?)\}", s))
    e = Counter(re.findall(r"\\end\{(\w+\*?)\}", s))
    mism = {k: (b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]}
    print("environment mismatches:", mism)
    labels = set(re.findall(r"\\label\{([^}]+)\}", s))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
    print("refs without label:", sorted(refs - labels))
    cites = {c.strip() for m in re.findall(r"\\cite[a-z]*(?:\[[^\]]*\])?\{([^}]+)\}", s) for c in m.split(",")}
    bib = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    print("citation keys missing from references.bib:", sorted(cites - keys))
    inputs = re.findall(r"\\input\{([^}]+)\}", s)
    missing_inputs = [i for i in inputs if not (PAPER_DIR / (i if i.endswith(".tex") else i + ".tex")).exists()]
    print("\\input files missing:", missing_inputs)
    files = [MAIN] + sorted(GENERATED_DIR.glob("tab_*.tex"))
    for p in files:
        txt = strip_comments(p.read_text(encoding="utf-8"))
        in_tab = False
        for i, ln in enumerate(txt.splitlines(), 1):
            if "\\begin{tabular" in ln:
                in_tab = True
            if in_tab and "&" in ln and not ln.rstrip().endswith("\\\\") and "\\caption" not in ln:
                print(f"row without \\\\ in {p.name}:{i}: {ln.strip()[:100]}")
                bad += 1
            if "\\end{tabular" in ln:
                in_tab = False
    ok = not (mism or (refs - labels) or (cites - keys) or missing_inputs or bad or s.count("{") != s.count("}"))
    print("LaTeX sanity:", "ok" if ok else "PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
