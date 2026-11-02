"""
Generate examples/act_three/REVIEW_RESPONSE.md from a template; every number is substituted from
paper/generated/macros.tex (so the letter and the paper cannot disagree).

    python -m examples.act_three.experiments.analysis.make_response
"""
import re
import sys

from .common import ACT_THREE_DIR, GENERATED_DIR

TEMPLATE = ACT_THREE_DIR / "experiments" / "analysis" / "response_template.md"
OUT = ACT_THREE_DIR / "REVIEW_RESPONSE.md"


def load_macros() -> dict:
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", (GENERATED_DIR / "macros.tex").read_text(encoding="utf-8"), re.M):
        out[m.group(1)] = m.group(2)
    return out


def plain(v: str) -> str:
    v = v.replace("$-$", "\u2212").replace("$+$", "+").replace("$<$", "<").replace("\\%", "%").replace("\\textbf{", "").replace("}", "")
    return v.replace("\\", "")


def main():
    mac = load_macros()
    text = TEMPLATE.read_text(encoding="utf-8")
    missing = []

    def sub(m):
        k = m.group(1)
        if k not in mac:
            missing.append(k)
            return "[[MISSING:" + k + "]]"
        return plain(mac[k])

    out = re.sub(r"\{\{([A-Za-z]+)\}\}", sub, text)
    OUT.write_text(out, encoding="utf-8")
    print(f"wrote {OUT}; missing macros: {sorted(set(missing))}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
