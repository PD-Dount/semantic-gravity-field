"""Normalise the two manuscripts to the md-to-word input contract and export .docx."""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
EXE = os.path.join(os.path.dirname(PY), "Scripts", "md-to-word.exe")

EN = dict(title="Title", subtitle="Subtitle", author="Author", abst_title="Abstract Title",
          abst="Abstract", h1="# ", h2="## ", img="Image Caption", tbl="Table Caption")
ZH = dict(title="文档标题", subtitle="副标题", author="作者", abst_title="摘要标题",
          abst="摘要", h1="# ", h2="## ", img="图片题注", tbl="表格题注")


def convert(src, dst, S, author, en=True):
    lines = open(src, encoding="utf-8").read().split("\n")
    out, i = [], 0
    tbl = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):                              # document title
            out.append(f"%% {S['title']} {ln[2:].strip()}")
            i += 1
            continue
        if ln.startswith("### ") and not any(x.startswith("# ") for x in out):
            out.append(f"%% {S['subtitle']} {ln[4:].strip()}")
            i += 1
            continue
        if ln.startswith("**作者") or ln.startswith("**Author"):
            out.append(f"%% {S['author']} {author}")
            i += 1
            continue
        if ln.startswith("**单位") or ln.startswith("**Affiliation") or ln.startswith("**Status"):
            out.append(f"%% {S['subtitle']} {re.sub(r'^\\*\\*[^*]+\\*\\*', '', ln).strip()}")
            i += 1
            continue
        if ln.startswith("## 摘要") or ln.startswith("## Abstract"):
            out.append(f"%% {S['abst_title']} {ln[3:].strip()}")
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and not lines[i].startswith("#"):
                out.append(f"%% {S['abst']} {lines[i].strip()}")
                i += 1
            continue
        if ln.startswith("## "):
            out.append("# " + ln[3:].strip())
            i += 1
            continue
        if ln.startswith("### "):
            out.append("## " + ln[4:].strip())
            i += 1
            continue
        m = re.match(r"^!\[.*?\]\((.+?)\)\s*$", ln)
        if m:
            out.append(f"![]({m.group(1)}){{width=14cm}}")
            i += 1
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("**Figure") or (j < len(lines) and lines[j].startswith("**图")):
                out.append("")
                out.append(f"%% {S['img']} {lines[j].strip()}")
                i = j + 1
            continue
        if re.fullmatch(r"-{3,}\s*", ln):            # horizontal rules are not supported
            i += 1
            continue
        if ln.startswith("|---") or ln.startswith("| ---"):
            tbl += 1
            out.append(ln)
            i += 1
            continue
        out.append(ln)
        i += 1
    open(dst, "w", encoding="utf-8").write("\n".join(out))
    return tbl


def out_has_section(out):
    return any(x.startswith("# ") or x.startswith("%% 文档标题") or x.startswith("%% Title") for x in out)


def run(md, docx, preset):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([EXE, md, "--preset", preset, "-o", docx],
                       cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    print(f"--- {os.path.basename(docx)} (exit {r.returncode}) ---")
    txt = (r.stdout or "") + (r.stderr or "")
    print(txt[-2000:])
    return r.returncode


if __name__ == "__main__":
    convert(os.path.join(HERE, "manuscript_en.md"), os.path.join(HERE, "_norm_en.md"), EN, "Zhang Junbin")
    convert(os.path.join(HERE, "manuscript_zh.md"), os.path.join(HERE, "_norm_zh.md"), ZH, "张俊滨")
    rc1 = run("_norm_en.md", "Semantic_Gravity_EN.docx", "en-nature")
    rc2 = run("_norm_zh.md", "语义引力场_中文.docx", "zh-academic")
    sys.exit(0 if rc1 == 0 and rc2 == 0 else 1)
