#!/usr/bin/env python3
"""Build the two back-matter files of the workbook FROM the chapters, so they cannot drift from them.

  notes/workbook/91-formulas.md   the formulas of one iteration on one page + every numbered equation (1)-(37), (G)
  notes/workbook/92-symbols.md    every row of every chapter's symbol table, with the chapter that teaches it

Nothing here is typed twice: an equation is copied from the `$$ ... \\tag{N} $$` line of the chapter that owns it
(with the name of its `> [!KEY]` box), and a symbol row is copied from the `| สัญลักษณ์ | ชนิด | ความหมาย |` table of
the first chapter that lists it. The only hand-written part is ONE_PAGE below, which only says WHEN each equation
runs and points at equation numbers.

Run from the repo root before scripts/workbook_to_pdf.py:   python3 scripts/workbook_backmatter.py
"""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / "notes/workbook"
ANSWER_MARK = "<!-- เฉลย -->"
NOTE = "<!-- สร้างโดย scripts/workbook_backmatter.py จากเนื้อหาของบทที่ 1-30 ห้ามแก้ไฟล์นี้ด้วยมือ ให้แก้ที่บทแล้วรันสคริปต์ใหม่ -->"

# (when, [equation tags in the order they run], what the step does). Tags must exist in the chapters.
ONE_PAGE = [
    ("ทุก step ตอนเก็บข้อมูล", ["9", "16"], "ประกอบระฆังแล้วสุ่ม action"),
    ("", ["14", "15"], "จด $\\log\\pi_{old}$ ของ action ที่สุ่มได้ ลงตาราง"),
    ("", ["17", "18"], "environment แปลง action เป็นมุมเป้าหมายและแรงบิด"),
    ("ครั้งเดียวต่อตาราง", ["19"], "critic ทำนายพลาดเท่าไรในหนึ่ง step"),
    ("", ["26"], "เดินย้อนจากแถวล่างขึ้นบน สะสมเป็น $A_t$"),
    ("", ["27"], "เป้าที่ใช้สอน critic"),
    ("", ["28"], "ปรับ $A_t$ ให้เทียบกันได้ทั้งตาราง"),
    # ppo.py:221-244 measures KL and moves alpha BEFORE the ratio at :252 (chapter 27, the five-step table)
    ("ทุก gradient step", ["30", "37"], "วัดว่าระฆังขยับไปเท่าไร แล้วปรับ $\\alpha$"),
    ("", ["21"], "ระฆังใหม่ชอบ action เดิมมากขึ้นกี่เท่า"),
    ("", ["29"], "loss ของ policy"),
    ("", ["32"], "loss ของ critic"),
    ("", ["31"], "ความกว้างของระฆัง"),
    ("", ["33"], "รวมสามพจน์เป็นตัวเลขตัวเดียว"),
    ("", ["G"], "ความชันที่ไหลถึง $\\mu$"),
    ("", ["35"], "ความชันที่ไหลถึง $\\ell$ จาก $L^{CLIP}$ และจาก $H$"),
    ("", ["36"], "Adam ก้าวลูกบิดทุกตัว"),
]


# Equations too tall for a table row are described in words there; their full form is in the list below the table.
WORDS = {"37": "ถ้า KL เฉลี่ยเกิน 0.02 หาร $\\alpha$ ด้วย 1.5 ถ้าต่ำกว่า 0.005 คูณด้วย 1.5 โดยอยู่ในช่วง $10^{-5}$ ถึง $10^{-2}$",
         "11": "ELU"}


def chapters():
    for f in sorted(SRC.glob("[0-9][0-9]-*.md")):
        n = int(f.name[:2])
        if 1 <= n <= 30 and "-0-part" not in f.name:
            yield n, f.read_text(encoding="utf-8").split(ANSWER_MARK)[0]


def equations():
    """{tag: (chapter, name of the KEY box, TeX without the tag)} for every `\\tag{...}` display in the chapters."""
    out = {}
    for n, text in chapters():
        key = ""
        for line in text.splitlines():
            m = re.match(r"^> \[!KEY\]\s*(.*)", line)
            if m:
                key = m.group(1).strip()
            m = re.search(r"\$\$(.*?)\\tag\{([^}]+)\}\s*\$\$", line)
            if m:
                tag = m.group(2)
                if tag in out:
                    raise SystemExit(f"equation ({tag}) is tagged twice: chapters {out[tag][0]} and {n}")
                out[tag] = (n, key, m.group(1).strip())
    return out


def atom(key):
    """What a symbol row is 'about': its first math chunk, else its first bold word, else its first code span."""
    for pat in (r"^\$[^$]+\$", r"\*\*`?([^*`]+)`?\*\*", r"`[^`]+`"):
        m = re.search(pat, key)
        if m:
            return re.sub(r"\s+", "", m.group(0))
    return re.sub(r"\s+", "", key)


def symbols():
    """[(first chapter, [later chapters], key cell, type cell, meaning cell)] in teaching order."""
    rows, index = [], {}
    head = re.compile(r"^\| *(?:สัญลักษณ์|คำ)[^|\n]*\| *ชนิด *\| *ความหมาย *\|[^\n]*\n\|[-| :]+\n((?:\|.*\n)+)", re.M)
    for n, text in chapters():
        for tm in head.finditer(text):
            for line in tm.group(1).strip().splitlines():
                cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
                if len(cells) < 3:
                    continue
                a = atom(cells[0])
                if a in index:
                    if n not in rows[index[a]][1] and n != rows[index[a]][0]:
                        rows[index[a]][1].append(n)
                else:
                    index[a] = len(rows)
                    rows.append([n, [], cells[0], cells[1], cells[2]])
    return rows


def main():
    eq = equations()
    order = sorted(eq, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else 0))
    missing = [t for _, tags, _ in ONE_PAGE for t in tags if t not in eq]
    if missing:
        raise SystemExit(f"ONE_PAGE names equations that no chapter tags: {missing}")

    f = [NOTE, "", "## สูตรทั้งหมดในหน้าเดียว", "",
         "หน้านี้รวมสูตรที่ใช้จริงในหนึ่ง iteration ของ PPO บน Tron1 เรียงตามลำดับที่โค้ดคำนวณ สูตรทุกตัวคัดมาจากบทที่สอน "
         "เลขในวงเล็บคือเลขประจำตัวของสมการตามบทนำ ถ้าเขียนสูตรเหล่านี้ได้ครบโดยไม่เปิดเล่ม และบอกได้ว่าแต่ละตัวทำงานเมื่อไร "
         "ก็เดิน iteration ของตารางตัวอย่างได้ด้วยมือ", "",
         "### หนึ่ง iteration เรียงตามเวลา",
         ]
    for when, tags, what in ONE_PAGE:
        if when:
            f += ["", f"**{when}**", "", "| สมการ | สูตร | ทำอะไร | บท |", "| --- | --- | --- | --- |"]
        for i, t in enumerate(tags):
            tex = eq[t][2] if t not in WORDS else None
            cell = f"${tex}$" if tex else WORDS[t]
            f.append(f"| ({t}) | {cell} | {what if i == 0 else ''} | {eq[t][0]} |")
    f += ["", "### สมการที่มีเลขทั้งหมด", "",
          f"เล่มนี้มีสมการที่มีเลข {len(eq)} ข้อ คือ (1) ถึง (37) และ (G) เรียงตามเลข ซึ่งไม่ใช่ลำดับที่พบในเล่ม", ""]
    for t in order:
        n, key, tex = eq[t]
        f += [f"**({t})** {key} · บทที่ {n}", "", f"$${tex}$$", ""]
    (SRC / "91-formulas.md").write_text("\n".join(f).rstrip() + "\n", encoding="utf-8")

    rows = symbols()
    s = [NOTE, "", "## ดัชนีสัญลักษณ์และคำ", "",
         "ตารางนี้รวมทุกแถวของตารางสัญลักษณ์ในหัวข้อ*ของที่ต้องมีก่อนเริ่ม*ของทุกบท เรียงตามบทที่สอน ถ้อยคำในคอลัมน์ชนิดและความหมาย"
         "คัดมาจากบทนั้นตรง ๆ เลขตัวแรกในคอลัมน์ขวาคือบทที่สอน เลขที่ตามมาคือบทที่นำกลับมาใช้ในตารางสัญลักษณ์ของบทนั้นอีกครั้ง", ""]
    # the part pages themselves say where each part starts and what it is called
    part_of = [(int(p.name[:2]), p.read_text(encoding="utf-8").strip().lstrip("#").strip())
               for p in sorted(SRC.glob("[0-9][0-9]-0-part.md")) if int(p.name[:2]) <= 30]
    bounds = [b for b, _ in part_of] + [99]
    for i, (lo, title) in enumerate(part_of):
        chunk = [r for r in rows if lo <= r[0] < bounds[i + 1]]
        if not chunk:
            continue
        s += [f"### {title}", "", "| สัญลักษณ์หรือคำ | ชนิด | ความหมาย | บท |", "| --- | --- | --- | --- |"]
        s += [f"| {k} | {ty} | {me} | {', '.join(map(str, [n] + later))} |" for n, later, k, ty, me in chunk]
        s.append("")
    (SRC / "92-symbols.md").write_text("\n".join(s).rstrip() + "\n", encoding="utf-8")
    print(f"91-formulas.md: {len(eq)} equations, {len(ONE_PAGE)} rows on the one-page table;  92-symbols.md: {len(rows)} rows")


if __name__ == "__main__":
    main()
