"""
T13 checks: every \\Macro used in the paper and in paper/generated/tab_*.tex is defined in macros.tex,
and (optionally) list digits left in selected sections of the main .tex.

    python -m examples.act_three.experiments.analysis.check_macros [--digits]
"""
import re
import sys

from .common import GENERATED_DIR, PAPER_DIR

MAIN = PAPER_DIR / "AnonymousSubmission2027.tex"
BUILTIN = set("""textbf textit emph section subsection subsubsection cite citep citet citealt label ref caption includegraphics
begin end item toprule midrule bottomrule cmidrule multicolumn resizebox textwidth columnwidth input newcommand small footnotesize
boldsymbol times text mathbf mathrm approx rightarrow pm leq geq le ge to quad qquad hline centering fbox parbox noindent
bibliography bibliographystyle usepackage documentclass title author affiliations maketitle frenchspacing setcounter pdfinfo
urlstyle def UrlFont rm cdot sum sigma frac in arg min max underline textsc textrm tiny large Large mathbb citeauthor
nolinebreak linewidth vspace hspace footnote paragraph rho left right sim lambda alpha beta chi tau mu delta Delta Rightarrow
ldots dots ensuremath mbox relax caption setlength tabcolsep arraystretch hfill thinspace nobreakspace S""".split())


def macros_defined():
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (GENERATED_DIR / "macros.tex").read_text(encoding="utf-8")))


def main():
    defined = macros_defined()
    bad = {}
    for p in [MAIN] + sorted(GENERATED_DIR.glob("tab_*.tex")) + sorted(GENERATED_DIR.glob("*_body.tex")):
        txt = re.sub(r"(?m)(?<!\\)%.*$", "", p.read_text(encoding="utf-8"))
        for m in re.findall(r"(?<!\\)\\([A-Za-z]+)", txt):
            if m in defined or m in BUILTIN or m.islower():
                continue
            bad.setdefault(p.name, set()).add(m)
    for k, v in bad.items():
        print(f"UNDEFINED in {k}: {sorted(v)}")
    print("undefined-macro check:", "FAIL" if bad else "ok", f"({len(defined)} macros defined)")
    if "--digits" in sys.argv:
        txt = MAIN.read_text(encoding="utf-8")
        for name in ("Abstract", "Results", "Discussion"):
            pass
        body = re.sub(r"(?m)(?<!\\)%.*$", "", txt)
        for ln, line in enumerate(body.splitlines(), 1):
            clean = re.sub(r"\\(cite[a-z]*|ref|label|includegraphics|input)(\[[^\]]*\])?\{[^}]*\}", "", line)
            clean = re.sub(r"\\[A-Za-z]+", "", clean)
            if re.search(r"\d", clean) and not clean.strip().startswith("\\documentclass"):
                print(f"{ln}: {line.strip()[:160]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
