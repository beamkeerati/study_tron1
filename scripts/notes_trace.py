#!/usr/bin/env python3
"""Work group 9 of the workbook: trace what work group 8 found in the book back into the source notes 00-07.

Work group 8 (commit 50dc919) fixed 165 places in notes/workbook/ and never looked at the notes the book was
written from.  Nothing here edits a note.  Every section prints candidates with `file:line`; the notes-reviewer
and a human decide which are defects.

  pairs      every line 50dc919 changed in a chapter, cut down to the words that changed (old -> new)
  families   notes 00-07 and diagrams/*.html searched for the claim families of work group 8
             (task 1 of the group-9 list in notes/workbook/_OUTLINE.md)
  six        the six last edits of work group 8 that were never re-reviewed (task 2): where they are now
  open       the items _HISTORY.md records as not fixed / not checked (task 3): the lines they point at
  glossary   the two self-contradictions of the glossary (task 4): every row that would have to move

usage:  python3 scripts/notes_trace.py [pairs|families|six|open|glossary] [--note 05]
"""
import difflib, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTES = sorted((ROOT / "notes").glob("0[0-7]-*.md")) + sorted((ROOT / "notes" / "diagrams").glob("*.html"))
BOOK = ROOT / "notes" / "workbook"
COMMIT = "50dc919"


def short(p):
    return p.name[:2] if p.suffix == ".md" else p.name


def cut(s, n):
    s = s.replace("\n", " // ")
    return s if len(s) <= n else s[:n] + "…"


def around(line, m, w=70):
    a, b = max(0, m.start() - w), min(len(line), m.end() + w)
    return ("…" if a else "") + line[a:m.start()] + "《" + m.group(0) + "》" + line[m.end():b] + ("…" if b < len(line) else "")


def section_of(lines, i):
    """nearest heading above line i (so a hit can be read as '05 §4.6')"""
    for j in range(i, -1, -1):
        if lines[j].startswith("#"):
            return lines[j].lstrip("# ").strip()[:46]
    return ""


# ---------------------------------------------------------------------------------------------- pairs
def pairs():
    diff = subprocess.run(["git", "show", COMMIT, "--format=", "-U0", "--", "notes/workbook/[0-3]*.md"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout
    tok = lambda s: re.findall(r"\$[^$]*\$|`[^`]*`|[A-Za-z0-9_.,+\-−]+|[\u0E00-\u0E7F]+|\s+|.", s)
    k = 0
    for block in re.split(r"(?m)^diff --git ", diff)[1:]:
        f = re.search(r"b/notes/workbook/(\S+)", block).group(1)
        for h in re.split(r"(?m)^@@ ", block)[1:]:
            head, _, body = h.partition("\n")
            ln = int(re.search(r"\+(\d+)", head).group(1))
            a = tok("\n".join(l[1:] for l in body.split("\n") if l.startswith("-")))
            b = tok("\n".join(l[1:] for l in body.split("\n") if l.startswith("+")))
            print(f"## {f}:{ln}")
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                if op == "equal":
                    continue
                k += 1
                print(f"  [{k}] …{cut(''.join(a[max(0, i1 - 14):i1])[-60:], 60)}《{cut(''.join(a[i1:i2]), 260)}》"
                      f"→《{cut(''.join(b[j1:j2]), 330)}》{cut(''.join(a[i2:i2 + 14]), 40)}…")
    print(f"\n{k} changed segments")


# ------------------------------------------------------------------------------------------- families
# (key, what work group 8 found in the book, patterns).  A pattern is a candidate finder, not a verdict.
FAMILIES = [
    ("C1 mass-through-encoder",
     "ch.29 said the policy guesses its own mass from the history h through the encoder; the actor only gets v-hat, "
     "which is trained to be the base velocity and nothing else (ch.21)",
     [r"(มวล|น้ำหนักตัว|แรงเสียดทาน|friction|mass)[^|\n]{0,160}(encoder|ประวัติ|\$h\$|obsHistory|history|เดา|ปรับตัว|adapt)",
      r"(encoder|ประวัติ|\$h\$|obsHistory|history)[^|\n]{0,160}(มวล|น้ำหนักตัว|แรงเสียดทาน|friction|mass|adapt|ปรับตัว|system identification|system id)"]),
    ("C2 critic-with-42",
     "ch.20 said a critic fed the actor's 42 values could not tell the two robots of ch.6 apart; the 42 include v-hat",
     [r"(หุ่น A|หุ่น B|สองตัว)[^\n]{0,200}(critic|แยก)", r"critic[^\n]{0,120}(แยกไม่|แยกไม่ออก|ให้ค่าเดียวกัน|เท่า actor|เหมือน actor)"]),
    ("C3 mask-is-1",
     "ch.22 derived GAE saying the mask is 1 in every cell; the last cell of an episode needs mask 0",
     [r"mask[^\n]{0,60}(เป็น 1|= ?1)[^\n]{0,40}(ทุก|ตลอด)", r"(ไม่มี done|ภายใน episode|episode เดียว)[^\n]{0,80}(mask|not_done|1 - d)",
      r"สมมติ[^\n]{0,60}(ไม่มี done|mask|ไม่ล้ม)", r"(ละ|ตัด|ทิ้ง|ไม่เขียน|ยังไม่ใส่)[^\n]{0,30}(mask|not_done|\(1 ?- ?d)"]),
    ("C4 actor-critic-independent",
     "ch.24 said actor and critic do not disturb each other; clip_grad_norm_ scales all their gradients by one number "
     "(ppo.py:289), ch.25",
     [r"ไม่(รบกวน|กวน|เกี่ยว|ปน|กระทบ|ส่งผล)[^\n]{0,30}กัน", r"(แยก|อิสระ)[^\n]{0,40}(actor|critic)[^\n]{0,60}(actor|critic)",
      r"(actor|critic)[^\n]{0,80}(อิสระ|เป็นอิสระ|แยกกันโดยสิ้นเชิง)", r"clip_grad_norm|max_grad_norm"]),
    ("F1 step-size-is-eq12",
     "statements about the SIZE of a knob's move that hold for equation (12) only; Tron1 steps with Adam, whose first "
     "step is alpha long whatever the gradient (ch.25)",
     [r"(ความชัน|gradient|\.grad)[^\n]{0,60}(ใหญ่|มาก|เล็ก|น้อย|เกือบศูนย์|ศูนย์)[^\n]{0,60}(ขยับ|ก้าว|หมุน|ปรับ|เปลี่ยน)[^\n]{0,20}(มาก|น้อย|ไกล|เยอะ|แทบไม่|ไม่)",
      r"(ขยับ|ก้าว|หมุน|ปรับ)[^\n]{0,30}(ตามสัดส่วน|เป็นสัดส่วน|proportional)",
      r"optimizer[^\n]{0,40}(สมการ|\(12\))", r"\(12\)[^\n]{0,60}(ppo\.py|optimizer|\.step\(\)|Adam)", r"(ppo\.py:290|optimizer\.step)[^\n]{0,80}\(12\)",
      r"vanishing|ความชันหาย"]),
    ("F2 alpha-constant",
     "alpha called fixed; rule (37) changes it every gradient step (ppo.py:221-235), 1e-3 is only where it starts",
     [r"(learning rate|\\alpha|`lr`|learning_rate)[^\n]{0,80}(คงที่|ไม่เปลี่ยน|ตายตัว|ตลอดการเทรน|fixed|constant)",
      r"(คงที่|ไม่เปลี่ยน|ตายตัว)[^\n]{0,60}(learning rate|\\alpha|hyperparameter)",
      r"hyperparameter[^\n]{0,80}(ไม่เปลี่ยน|คงที่|ไม่ถูกปรับ|ไม่ได้เรียน|ตั้งครั้งเดียว)",
      r"(ใช้|=|คือ|เป็น) ?\$?(10\^\{-3\}|1e-3|0\.001)\$?", r"learning_rate ?= ?1\.?0?e-3"]),
    ("F3 on-the-real-robot",
     "'on the real robot' used for where a program or a network runs; the deploy code talks to the robot over the "
     "network and nothing in the snapshot says which machine it runs on (main.py:28-36)",
     [r"บนหุ่น", r"ลงหุ่น", r"ไปกับหุ่น", r"ในตัวหุ่น", r"onboard|on the robot|on the real robot|on-robot|on robot", r"บนตัวหุ่น|ฝังใน"]),
    ("F4 deploy-values",
     "values of the deploy layer assumed equal to training: kd 3.0 vs 1.5, ankle torque limit, hip default +-0.13, "
     "loop 500 Hz vs policy 50 Hz, PD computed on the receiving side",
     [r"1 ?kHz|1000 ?Hz|500 ?Hz", r"limxsdk|libpython|4\.0\.2", r"(deploy|หุ่นจริง|real robot)[^\n]{0,120}(เหมือน|เท่า|เดียวกัน|same|identical|unchanged)",
      r"(เหมือน|เท่า|เดียวกัน|same|identical)[^\n]{0,80}(deploy|หุ่นจริง|real robot)",
      r"effort_limit|80 ?N|สเปก", r"(stiffness|damping|k_p|k_d|Kp|Kd)[^\n]{0,60}(deploy|หุ่นจริง|real|SDK)",
      r"(SDK|หุ่นจริง|real robot|deploy)[^\n]{0,80}(stiffness|damping|k_p|k_d|Kp|Kd|PD|แรงบิด|torque)"]),
    ("X1 reset-pose",
     "an episode 'starts in the initial pose'; reset randomises position, joint angles (+-0.2 rad) and yaw over the "
     "full circle (limx_base_env_cfg.py:311-333)",
     [r"(กลับ|วาง|เริ่ม)[^\n]{0,40}(ท่าเริ่มต้น|ท่าตั้งต้น|ท่ายืน|initial pose|default pose)", r"reset[^\n]{0,80}(ท่า|pose)"]),
    ("X2 trained-with-2048",
     "'trained with 2048 envs': exp_001 has not been run; the real logs are 16 and 64 envs",
     [r"(เทรน|train|รัน)[^\n]{0,30}(ด้วย|with)?[^\n]{0,20}2048", r"exp_001", r"(การรันจริง|ล็อกจริง|real run|ที่วัดได้|measured)"]),
    ("X3 critic-target-is-reality",
     "the critic's error called 'prediction minus what really happened'; the target R_t is built from real rewards "
     "PLUS the critic's own predictions",
     [r"(สิ่งที่เกิดจริง|ผลจริง|ความจริง|ของจริง|ค่าจริง)[^\n]{0,80}(critic|R_t|return|\\hat R|target|เฉลย|เป้า)",
      r"(critic|R_t|returns|เฉลย|เป้า)[^\n]{0,80}(สิ่งที่เกิดจริง|ผลจริง|เกิดขึ้นจริง|ความจริง)"]),
    ("X4 sensors-cannot-give",
     "critic-only observations called 'what no sensor on the robot can measure'; safe wording: what the actor is not given",
     [r"เซนเซอร์[^\n]{0,80}(ไม่ให้|ไม่มี|วัดไม่ได้|ไม่ได้)", r"(ไม่มี|วัดไม่ได้)[^\n]{0,40}เซนเซอร์", r"privileged"]),
    ("X5 the-only-random-point",
     "E3 called the only place in the chain with sampling; E11 calls the same function and discards the sample, and "
     "observation noise is added on the environment side",
     [r"(จุดเดียว|ที่เดียว|ครั้งเดียว|แห่งเดียว)[^\n]{0,60}(สุ่ม|sample|random)", r"(สุ่ม|sample)[^\n]{0,60}(จุดเดียว|ที่เดียว|ครั้งเดียว|แห่งเดียว)"]),
    ("X6 clip-forbids",
     "clip / rule (37) / clip_grad_norm_ described as a wall that keeps or forbids the new policy from moving away",
     [r"(clip|KL|desired_kl|trust)[^\n]{0,80}(กันไม่ให้|ห้าม|บังคับ|การันตี|รับประกัน|ไม่มีทาง|ไม่ให้)",
      r"(กันไม่ให้|ห้าม|บังคับ|การันตี|รับประกัน)[^\n]{0,80}(policy|\\rho|\\pi|ระฆัง)[^\n]{0,60}(ไกล|เกิน|หลุด|เปลี่ยน)"]),
    ("X7 inference-is-calmer",
     "the robot under act_inference 'always walks more steadily' / 'a robot that samples would shake': the book could "
     "only support 'the target angle is not shifted by sigma each step'",
     [r"(สั่น|นิ่งกว่า|กระตุก|เดินนิ่ง|ส่าย|jitter|shak)", r"act_inference[^\n]{0,120}(เสมอ|ดีกว่า|นิ่ง)"]),
    ("X8 counts",
     "counts that later chapters corrected: CriticCfg 21 in the base class, 20 for SF (14 -> 13 critic-only); encoder "
     "loop ppo.py:299-324; 469,905 = actor + critic + logstd only; 833 gradient steps per 1000 env steps (not 840)",
     [r"\b21 ?(รายการ|terms|term|ตัว|obs)", r"\b14 ?(รายการ|terms)", r"ppo\.py:299-3\d\d", r"469,?905", r"\b840\b|\b833\b",
      r"`?model_\w+\.pt`?[^\n]{0,80}(policy|ตัวควบคุม|controller|weights)"]),
    ("X9 pd-rate-and-config-read",
     "PD 'runs four times faster than the policy' (true in the simulator only); a config value 'is used at once' "
     "(kp/kd are multiplied by a random factor at reset, events.py)",
     [r"(สี่เท่า|4 ?เท่า|4x|four times|200 ?Hz)", r"(stiffness|damping|kp|kd|k_p|k_d)[^\n]{0,60}(40|45|2\.5|1\.5|3\.0|50\.0)"]),
    ("X10 three-signals-only",
     "agent and environment 'talk through three signals only'; env.step returns five things (ch.3)",
     [r"(สามเส้น|สามอย่าง|3 อย่าง|สามสิ่ง)[^\n]{0,40}(เท่านั้น|only)", r"(เท่านั้น)[^\n]{0,30}(ข้ามเส้น|ระหว่าง agent)"]),
]


def unwrap(lines):
    """Notes 05-07 are hard-wrapped at ~80 columns, so a claim is split over lines.  Join the lines of one paragraph
    with a space (same length as the newline, so offsets still map to line numbers); table rows, headings and code
    fences stay apart."""
    out, starts, pos, fence = [], [], 0, False
    for i, l in enumerate(lines):
        starts.append(pos)
        pos += len(l) + 1
        if l.lstrip().startswith("```"):
            fence = not fence
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        hard = (fence or not l.strip() or not nxt.strip() or l.startswith(("|", "#")) or nxt.startswith(("|", "#"))
                or nxt.lstrip().startswith(("```", "- ", "* ")) or re.match(r"\s*\d+\. ", nxt) or l.lstrip().startswith("```"))
        out.append(l + ("\n" if hard else " "))
    return "".join(out), starts


def families(only=None):
    import bisect
    tot = {}
    for key, what, pats in FAMILIES:
        print("=" * 110, f"\n{key}\n  {what}\n" + "=" * 110)
        for p in NOTES:
            if only and not p.name.startswith(only):
                continue
            lines = p.read_text(encoding="utf-8").split("\n")
            text, starts = unwrap(lines)
            seen = set()
            for pat in pats:
                for m in re.finditer(pat, text, re.I):
                    i = bisect.bisect_right(starts, m.start()) - 1
                    if (i, (m.start() - starts[i]) // 60) in seen:
                        continue
                    seen.add((i, (m.start() - starts[i]) // 60))
                    a = text.rfind("\n", 0, m.start()) + 1
                    b = text.find("\n", m.end())
                    print(f"  {short(p)}:{i + 1:<5} [{section_of(lines, i)}]  {around(text[a:b], _M(m, a))}")
                    tot[(key, short(p))] = tot.get((key, short(p)), 0) + 1
    print("\n" + "=" * 110 + "\ncandidates per family and note\n" + "=" * 110)
    cols = sorted({n for _, n in tot})
    print(f"{'':30}" + "".join(f"{c[:11]:>12}" for c in cols))
    for key, _, _ in FAMILIES:
        print(f"{key:30}" + "".join(f"{tot.get((key, c), 0):>12}" for c in cols))


class _M:
    """a match re-based to the start of its paragraph, for around()"""
    def __init__(self, m, a):
        self._s, self._e, self._g = m.start() - a, m.end() - a, m.group(0)
    start = lambda self: self._s
    end = lambda self: self._e
    group = lambda self, n=0: self._g


# ------------------------------------------------------------------------------------------------ six
SIX = [("07", "ทิศที่หุ่นหันถูกสุ่มได้รอบตัว"), ("18", "สองข้อจากสามข้อ"), ("18", "บทที่ 26 เพิ่มกฎ"),
       ("24", "หนึ่งในนั้นคือบรรทัดที่ย่อความชัน"), ("28", "สถานีเดียวในทั้งสายที่ผลของการสุ่มถูกใช้ต่อ"),
       ("29", "มีแนวโน้มได้คะแนนเฉลี่ยต่ำลง")]


def six():
    for ch, text in SIX:
        f = next(BOOK.glob(f"{ch}-[!0]*.md"))
        hits = [(i + 1, l) for i, l in enumerate(f.read_text(encoding="utf-8").split("\n")) if text in l]
        print("=" * 110, f"\nch.{ch}  \"{text}\"  -> {len(hits)} hit(s)\n" + "=" * 110)
        for i, l in hits:
            print(f"{f.name}:{i}\n  {l}\n")


# ----------------------------------------------------------------------------------------------- open
OPEN = [("05", r"output_detach", "05 §4.4 Remark: output_detach is given as THE reason the policy loss does not move psi"),
        ("05", r"ppo\.py:144|ppo\.py:177|`:177`|`:144`", "05 §4.6 (ง): .detach() at ppo.py:144 / :177 'stops V in A-hat and R from carrying gradient' - never checked"),
        ("05", r"81920|0\.420|0\.180", "05 §4.7: the Computation line is an invented number and the note does not say so (07 does)"),
        ("07", r"81920|0\.420|0\.180", "07 §6: the same line, labelled"),
        ("05", r"24 ?> ?17|\b23 ?%|ร้อยละ 23|\b17\b[^\n]{0,30}step", "05 §4.7: '24 > 17' and 23% are the top row only; averaged over all rows it is 49%"),
        ("02", r"kHz|limxsdk|libpython|4\.0\.2|3\.8", "02: 1 kHz, limxsdk 4.0.2, libpython3.8 - never checked against the deploy snapshot"),
        ]


def open_items():
    for note, pat, what in OPEN:
        p = next((ROOT / "notes").glob(f"{note}-*.md"))
        lines = p.read_text(encoding="utf-8").split("\n")
        print("=" * 110, f"\n{what}\n" + "=" * 110)
        for i, line in enumerate(lines):
            m = re.search(pat, line)
            if m:
                print(f"  {note}:{i + 1:<5} [{section_of(lines, i)}]  {around(line, m, 110)}")
    f = ROOT / "scripts" / "notes_to_pdf.py"
    print("=" * 110, "\nscripts/notes_to_pdf.py: bold-looking math and Thai inside \\text{} (how it typesets math)\n" + "=" * 110)
    for i, line in enumerate(f.read_text(encoding="utf-8").split("\n")):
        if re.search(r"mathjax|katex|latex|weasy|chrome|font-weight|\\\\text|svg|chtml", line, re.I):
            print(f"  notes_to_pdf.py:{i + 1:<4} {line.strip()[:150]}")
    n = 0
    for p in NOTES:
        if p.suffix != ".md":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").split("\n")):
            for m in re.finditer(r"\\text\{[^}]*[\u0E00-\u0E7F][^}]*\}", line):
                n += 1
                if n <= 12:
                    print(f"  Thai in \\text: {short(p)}:{i + 1}  {m.group(0)[:60]}")
    print(f"  Thai inside \\text{{}} in notes 00-07: {n} place(s)")


# ------------------------------------------------------------------------------------------- glossary
def glossary():
    g = (ROOT / "notes" / "00-glossary.md").read_text(encoding="utf-8").split("\n")
    print("=" * 110, "\n(4a) theta: 'ตัวเลขทั้งหมดที่การเรียนรู้ไปเปลี่ยน' - and every other place that wording lives\n" + "=" * 110)
    for p in NOTES + sorted(BOOK.glob("[0-9]*.md")):
        for i, line in enumerate(p.read_text(encoding="utf-8").split("\n")):
            m = re.search(r"ตัวเลขทั้งหมดที่การเรียนรู้|ตัวเลขทุกตัวที่การเรียนรู้|ทั้งหมดที่การเรียนรู้ไปเปลี่ยน", line)
            if m:
                print(f"  {p.name[:26]}:{i + 1:<5} {around(line, m, 120)}")
    print("\n  rows of the glossary for theta, phi, psi and 'parameter':")
    for i, line in enumerate(g):
        if re.match(r"\| ?\**\$?\\(theta|phi|psi)\b", line) or re.match(r"\| ?\**(parameter|weight)", line, re.I):
            print(f"  00:{i + 1:<5} {cut(line, 330)}")
    print("=" * 110, "\n(4b) A and R without a subscript or brackets (glossary §7 forbids 'A ลอย ๆ', 'R ลอย ๆ'): count per file\n"
          "     A, R = the letter alone · hat A = normalised advantage without t · agg(A) = inside mean()/std() · bar A\n" + "=" * 110)
    import collections
    pat = re.compile(r"(\\hat\s*\{?\s*|\\bar\s*\{?\s*|\\mathcal\s*\{?\s*|\\mathbb\s*\{?\s*|\\mathbf\s*\{?\s*)?(?<![A-Za-z\\_])([AR])\}?(?![A-Za-z])")
    cnt, ex = collections.Counter(), collections.defaultdict(list)
    for p in [q for q in NOTES if q.suffix == ".md"] + sorted(BOOK.glob("[0-9]*.md")):
        fence = False
        for i, line in enumerate(p.read_text(encoding="utf-8").split("\n")):
            if line.lstrip().startswith("```"):
                fence = not fence
            if fence:
                continue
            for mm in re.finditer(r"\$\$?([^$]+)\$\$?", line):
                body = re.sub(r"\\(text|operatorname|mathrm|texttt|tag)\{[^}]*\}", lambda m: " " * len(m.group(0)), mm.group(1))
                for m in pat.finditer(body):
                    pre = m.group(1) or ""
                    if re.search("mathcal|mathbb|mathbf", pre) or body[m.end():].lstrip()[:1] in ("_", "^", "(", "'"):
                        continue
                    cls = ("hat " if "hat" in pre else "bar " if "bar" in pre else "") + m.group(2)
                    if re.search(r"(mean|std)(\}|_\{?n-1\}?)*\s*\(\s*$", mm.group(1)[:m.start()]):
                        cls = "agg(" + cls + ")"
                    cnt[(p.name, cls)] += 1
                    ex[cls].append(f"{p.name[:24]}:{i + 1:<5} ${cut(mm.group(1), 90)}$")
    names = sorted({k[0] for k in cnt})
    classes = [c for c in ("A", "R", "hat A", "bar A", "agg(A)") if any(cnt[(n, c)] for n in names)]
    print(f"{'':34}" + "".join(f"{c:>9}" for c in classes))
    for n in names:
        print(f"{n[:33]:34}" + "".join(f"{cnt[(n, c)]:>9}" for c in classes))
    print(f"{'TOTAL':34}" + "".join(f"{sum(cnt[(n, c)] for n in names):>9}" for c in classes))
    for c in ("agg(A)", "bar A", "R", "A"):
        print(f"\n  -- {c}")
        for e in ex[c]:
            print("  " + e)
    print("\n  the rule itself, and the row of equation (28):")
    for i, line in enumerate(g):
        if ("ลอย" in line) or re.search(r"\(28\)", line) and line.startswith("|"):
            print(f"  00:{i + 1:<5} {cut(line, 420)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else None
    what = args[0] if args else "all"
    if what in ("pairs",):
        pairs()
    if what in ("families", "all"):
        families(only)
    if what in ("six", "all"):
        six()
    if what in ("open", "all"):
        open_items()
    if what in ("glossary", "all"):
        glossary()
