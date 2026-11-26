#!/usr/bin/env python3
"""Work group 8 of the workbook: the checks that only make sense on the whole book at once.

Nothing here edits a chapter. Every section prints candidates with `file:line`; a human (or the notes-reviewer)
decides which are defects. The sections follow the task list of work group 8 (now in notes/workbook/_HISTORY.md):

  terms        (1) a term from the outline's "คำใหม่ที่สอน" column used by an EARLIER chapter than the one that
                   teaches it, + numbered equations used before their owner, + the outline column vs the symbol tables
  refs         (2) every sentence that names another chapter ("บทที่ N บอกว่า…", "บทที่ N จะ…"), with the mechanical
                   part checked here: the chapter exists, the tense matches the direction, a cited section /
                   figure / exercise number exists.  The list is what the reviewer reads.
  numbers      (4) every number that appears in two or more chapters: is it printed by the answer key, and is there
                   a near-twin (same value, different rounding) elsewhere in the book
  conventions  (5) the word agreements of _OUTLINE.md ("ข้อตกลงเรื่องคำ"), as (pattern, chapters where it must not be)
                   + (7) no "ภาค" + (8) no Thai digits
  dots         (6) stray ink in the bottom margin of the built PDF (the tone-mark that leaked across a page break)

Usage (repo root):  python3 scripts/workbook_audit.py [terms|refs|numbers|conventions|dots|all] [--part N] [--out DIR]
"""
import collections
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import workbook_backmatter as bm  # noqa: E402  (chapters(), equations(), symbols(), atom())

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "notes/workbook"
PDF = REPO / "notes/pdf/tron1-ppo-workbook.pdf"
ANSWER_MARK = "<!-- เฉลย -->"
PARTS = [(1, 1, 2), (2, 3, 9), (3, 10, 15), (4, 16, 18), (5, 19, 27), (6, 28, 30)]   # (part, first, last chapter)


def part_of(ch):
    return next((p for p, a, b in PARTS if a <= ch <= b), 0)


def files():
    """[(chapter number or 0 for the preface, path, full text INCLUDING the answers)] in page order."""
    out = [(0, SRC / "00-preface.md", (SRC / "00-preface.md").read_text(encoding="utf-8"))]
    for f in sorted(SRC.glob("[0-9][0-9]-*.md")):
        n = int(f.name[:2])
        if 1 <= n <= 30 and "-0-part" not in f.name:
            out.append((n, f, f.read_text(encoding="utf-8")))
    return out


def split_kinds(line):
    """A line -> (prose with code and math blanked, [math chunks], [code spans])."""
    code = re.findall(r"`[^`]*`", line)
    rest = re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), line)
    math = re.findall(r"\$\$.*?\$\$|\$[^$]+\$", rest)
    prose = re.sub(r"\$\$.*?\$\$|\$[^$]+\$", lambda m: " " * len(m.group(0)), rest)
    return prose, math, code


def lines_of(text):
    """(line number, line, in_fence, in_answers) for every line."""
    fence = ans = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if ANSWER_MARK in line:
            ans = True
        yield i, line, fence, ans


# =============================================================================================== (1) terms
def outline_rows():
    """{chapter: (title, raw 'คำใหม่ที่สอน' cell, [prerequisite chapters])} from the per-chapter table of _OUTLINE.md."""
    rows = {}
    for line in (SRC / "_OUTLINE.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) >= 6 and re.fullmatch(r"\d+", cells[0]):
            rows[int(cells[0])] = (cells[1], cells[2], [int(x) for x in re.findall(r"\d+", cells[3])])
    return rows


# Latin words in the outline column that are not terms of art (connectives, file names, units, names of other things).
NOT_A_TERM = set("""Tron1 TRON1 LimX PF SF WF N B T ROS path config obs ID PD Hz MB s the of vs
    step env policy reward action state toy e0 t0 mb PyTorch GPU CPU Isaac Lab Sim PPO RL""".split())
# Terms the column names only in Thai; (chapter, regex over prose). Added by hand from the column, nothing else.
THAI_TERMS = [
    (4, r"ตัวแปรสุ่ม"), (4, r"การแจกแจง"), (4, r"ค่าคาดหวัง"), (8, r"ระยะมอง"), (10, r"ความหนาแน่น"),
    (10, r"ระฆัง"), (25, r"ค่าเฉลี่ยเคลื่อนที่"), (26, r"ขั้นบันได"), (28, r"สถานี"), (16, r"รอบกวาด"),
    (23, r"หนีบ"), (23, r"พจน์ดิบ"), (23, r"คู่แฝด"), (25, r"ลูกบิดสามตัว"), (29, r"ค่าตั้งต้น"),
]
# Latin terms that the column gives in a longer phrase; (chapter, regex over prose, label).
EXTRA_LATIN = [
    (3, r"\bagent\b", "agent"), (3, r"\benvironment\b", "environment"), (3, r"\breward\b", "reward"),
    (3, r"\baction\b", "action"), (3, r"\bpolicy\b", "policy"), (3, r"\bRL\b", "RL"),
    (7, r"\bstep\b", "step"), (7, r"(?<!conda )\benv\b", "env"), (7, r"\bepisode\b", "episode"), (7, r"\bdone\b", "done"),
    (7, r"\brollout\b", "rollout"), (7, r"\biteration\b", "iteration"), (7, r"\btermination\b", "termination"),
    (7, r"\btruncation\b", "truncation"), (7, r"\btransition\b", "transition (unit)"),
    (8, r"\breturn\b", "return"), (8, r"\bobjective\b", "objective"),
    (9, r"\bcritic\b", "critic"), (9, r"\bbootstrap", "bootstrapping"), (9, r"\badvantage\b", "advantage"),
    (9, r"\bvalue function\b", "value function"), (9, r"\bBellman\b", "Bellman"),
    (5, r"\bMDP\b", "MDP"), (5, r"\bobservation\b", "observation"), (5, r"\bcommand\b", "command"),
    (6, r"\bMarkov\b", "Markov"), (6, r"\bPOMDP\b", "POMDP"),
    (10, r"\bGaussian\b", "Gaussian"), (10, r"\bPDF\b(?! ของ| ฉบับ)", "PDF (density)"), (10, r"\blog\b", "log"),
    (11, r"\bstochastic\b", "stochastic"), (11, r"\bdeterministic\b", "deterministic"),
    (12, r"\bexplor", "explore"), (12, r"\bexploit", "exploit"), (12, r"\bnoise\b", "noise"),
    (13, r"\bweight\b", "weight"), (13, r"\bbias\b", "bias"), (13, r"\blayer\b", "layer"), (13, r"\bELU\b", "ELU"),
    (13, r"\bMLP\b", "MLP"), (13, r"\bparameter\b", "parameter"), (13, r"\bnode\b", "node"),
    (13, r"\bhyperparameter", "hyperparameter"), (13, r"\bactivation\b", "activation"),
    (14, r"\bloss\b", "loss"), (14, r"\bgradient\b", "gradient"), (14, r"\bSGD\b", "SGD"),
    (14, r"\blearning rate\b", "learning rate"),
    (16, r"\btabular\b", "tabular"), (16, r"\bDP\b", "DP"), (16, r"\bfunction approximator\b", "function approximator"),
    (17, r"\bMonte Carlo\b|\bMC\b", "MC"), (17, r"\bTD\b", "TD"), (17, r"\bmask\b", "mask"),
    (17, r"\bvariance\b", "variance"),
    (18, r"\bon-policy\b", "on-policy"), (18, r"\boff-policy\b", "off-policy"),
    (18, r"\bimportance sampling\b", "importance sampling"),
    (19, r"\bpolicy gradient\b", "policy gradient"), (19, r"\bscore function\b", "score function"),
    (19, r"\bREINFORCE\b", "REINFORCE"),
    (20, r"\bactor\b", "actor"), (20, r"\basymmetric\b", "asymmetric"), (20, r"\bactor-critic\b", "actor-critic"),
    (21, r"\bencoder\b", "encoder"), (21, r"\bsupervised\b", "supervised learning"), (21, r"\bMSE\b", "MSE"),
    (21, r"\bdetach", "detach"), (21, r"\blatent\b", "latent"),
    (22, r"\bGAE\b", "GAE"),
    (23, r"\bsurrogate\b", "surrogate"), (23, r"\bclip\b", "clip"), (23, r"\bProximal\b", "PPO full name"),
    (24, r"\bentropy\b", "entropy"), (24, r"\bvalue loss\b", "value loss"),
    (25, r"\boptimizer\b", "optimizer"), (25, r"\bAdam\b", "Adam"), (25, r"\bnorm\b", "norm"),
    (26, r"\bKL\b", "KL"),
    (27, r"\bmini-batch\b", "mini-batch"), (27, r"\bepoch\b", "epoch"),
    (29, r"\bdomain randomi[sz]ation\b", "domain randomisation"), (29, r"\bcurriculum\b", "curriculum"),
    (30, r"\bONNX\b", "ONNX"), (30, r"\bSDK\b", "SDK"), (30, r"\bserver mode\b", "server mode"),
    (2, r"\btensor\b", "tensor"), (2, r"\bregistry\b", "registry"), (2, r"\bmanager\b", "manager"),
    (2, r"\btask ID\b", "task ID"),
]
# Symbols: (chapter that teaches it, regex over a math chunk, label). From the column's `$…$` cells.
MATH_TERMS = [
    (4, r"\\mathbb\{?E\}?", r"\mathbb{E}"), (4, r"\\sim", r"\sim"), (4, r"\\mid", r"\mid"),
    (5, r"\\mathcal\{?S\}?", r"\mathcal{S}"), (5, r"\\mathcal\{?A\}?", r"\mathcal{A}"), (5, r"\\gamma", r"\gamma"),
    (5, r"s_t|s_\{t", "s_t"), (5, r"a_t|a_\{t", "a_t"), (5, r"r_t|r_\{t", "r_t"),
    (8, r"G_t|G_\{t", "G_t"), (8, r"\\arg\\max", r"\arg\max"), (8, r"\\pi\^\*|\\pi\^\{\*\}", r"\pi^*"),
    (8, r"(?<![a-zA-Z\\])J(?![a-zA-Z])", "J"),
    (9, r"V\^\{?\\pi", r"V^\pi"), (9, r"V_\{?last", "V_last"), (9, r"Q\^\{?\\pi|(?<![a-zA-Z\\])Q\(", r"Q"),
    (9, r"A\^\{?\\pi", r"A^\pi"),
    (10, r"\\mu", r"\mu"), (10, r"\\sigma", r"\sigma"),
    (11, r"\\pi_\{?\\theta", r"\pi_\theta"), (11, r"\\theta", r"\theta"), (11, r"\\ell", r"\ell"),
    (11, r"\\Delta", r"\Delta"),
    (12, r"\\varepsilon", r"\varepsilon"),
    (14, r"\\nabla", r"\nabla"), (14, r"\\alpha", r"\alpha"), (14, r"(?<![a-zA-Z\\_^{])L(?![a-zA-Z])", "L (loss)"),
    (17, r"\\delta", r"\delta"),
    (18, r"\\rho", r"\rho"), (18, r"\\pi_\{?(old|new)", r"\pi_old / \pi_new"),
    (20, r"\\phi", r"\phi"),
    (21, r"\\psi", r"\psi"), (21, r"\\hat\{? ?v\}?(?!_)", r"\hat v (latent)"),
    (22, r"\\lambda", r"\lambda"), (22, r"\\hat\{? ?A\}?", r"\hat A"), (22, r"(?<![a-zA-Z\\])R_t|R_\{t", "R_t"),
    (22, r"(?<![a-zA-Z\\^])A_t|(?<![a-zA-Z\\^])A_\{t", "A_t"),
    (23, r"\\epsilon", r"\epsilon"), (23, r"L\^\{?CLIP", "L^CLIP"), (23, r"L_i", "L_i"),
    (24, r"V_\{?(old|new|clip)", "V_old / V_new / V_clip"), (24, r"L_V|L_\{V", "L_V"), (24, r"c_V|c_H", "c_V, c_H"),
    (24, r"(?<![a-zA-Z\\])H(?![a-zA-Z])", "H"),
    (25, r"\\beta", r"\beta"), (25, r"(?<![a-z])[mvg]_k|\\hat\{? ?[mv]\}?_k", "m_k, v_k, g_k"), (25, r"b_V", "b_V"),
    (26, r"\\mu_o|\\sigma_o|\\mu_n|\\sigma_n", r"\mu_o …"), (26, r"D_\{?KL|\\mathrm\{KL\}|KL", "KL"),
]


def terms():
    rows = outline_rows()
    fs = files()
    print("=" * 100, "\n(1a) terms used before the chapter that teaches them\n" + "=" * 100)
    print("A line is a candidate when a term taught in chapter K appears in a chapter < K. `announced` means the same\n"
          "line names chapter K (a pointer forward, which _STYLE.md asks to avoid or to say in plain words).\n")
    hits = collections.defaultdict(list)                 # (label, K) -> [(chapter, line no, announced, where)]
    for n, f, text in fs:
        for i, line, fence, ans in lines_of(text):
            if fence:
                continue
            prose, math, _ = split_kinds(line)
            later = {int(x) for x in re.findall(r"บทที่ (\d+)", line)}
            for k, pat, label in EXTRA_LATIN:
                if n < k and re.search(pat, prose):
                    hits[(label, k)].append((n, i, k in later, "prose"))
            for k, pat in THAI_TERMS:
                if n < k and re.search(pat, prose):
                    hits[(pat, k)].append((n, i, k in later, "prose"))
            for k, pat, label in MATH_TERMS:
                if n < k and any(re.search(pat, m) for m in math):
                    hits[("$" + label + "$", k)].append((n, i, k in later, "math"))
    name = {n: f.name for n, f, _ in fs}
    print(f"  {len(EXTRA_LATIN) + len(THAI_TERMS) + len(MATH_TERMS)} terms checked "
          f"({len(EXTRA_LATIN)} Latin, {len(THAI_TERMS)} Thai, {len(MATH_TERMS)} symbols); {len(hits)} have an earlier use:\n")
    for (label, k), hs in sorted(hits.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        bare = [h for h in hs if not h[2]]
        per = collections.Counter(h[0] for h in hs)
        print(f"  {label}  (taught in ch {k}): {len(hs)} line(s) earlier, {len(bare)} without a pointer to ch {k}"
              f"   chapters {dict(sorted(per.items()))}")
        for n, i, ann, _ in hs[:6]:
            print(f"      {name[n]}:{i}{'  announced' if ann else ''}")
        if len(hs) > 6:
            print(f"      … {len(hs) - 6} more")

    print("\n" + "=" * 100, "\n(1b) numbered equations cited before the chapter that owns them\n" + "=" * 100)
    eq = bm.equations()
    for n, f, text in fs:
        for i, line, fence, ans in lines_of(text):
            for m in re.finditer(r"สมการ \((\d+|G)\)(?:[–-]\((\d+)\))?", line):
                tags = [m.group(1)] + ([m.group(2)] if m.group(2) else [])
                for t in tags:
                    if t in eq and eq[t][0] > n:
                        print(f"  {f.name}:{i}  cites ({t}), owned by chapter {eq[t][0]}:  {line.strip()[:150]}")
                    if t not in eq:
                        print(f"  {f.name}:{i}  cites ({t}) which no chapter tags")
    owners = collections.defaultdict(list)
    for t, (n, key, _) in eq.items():
        owners[n].append(t)
    print("\n  equation owners vs the outline column:")
    for n in sorted(owners):
        col = rows[n][1]
        claimed = set()
        for a, b in re.findall(r"\((\d+)\)[–-]\((\d+)\)", col):
            claimed |= {str(x) for x in range(int(a), int(b) + 1)}
        claimed |= set(re.findall(r"\((\d+|G)\)", col))
        got = set(owners[n])
        if claimed != got:
            print(f"    chapter {n}: column says {sorted(claimed)}, chapter tags {sorted(got)}"
                  f"  (only in column {sorted(claimed - got)}, only in chapter {sorted(got - claimed)})")

    print("\n" + "=" * 100, "\n(1c) the outline column vs the chapters' symbol tables\n" + "=" * 100)
    print("A row of a chapter's symbol table whose key does not occur in that chapter's 'คำใหม่ที่สอน' cell. The\n"
          "column is what later sessions read to know what has been taught, so a row missing there can be re-taught.\n")
    for first, later, key, ty, me in bm.symbols():
        col = rows.get(first, ("", "", []))[1]
        a = bm.atom(key)
        words = re.findall(r"[A-Za-z][A-Za-z_-]{2,}", re.sub(r"\$[^$]*\$", " ", key))
        thai = re.sub(r"[`*$]", "", key).strip()
        seen = (a in re.sub(r"\s+", "", col)) or any(w.lower() in col.lower() for w in words) or (thai and thai in col)
        if not seen:
            print(f"  ch {first:2d}  {key[:70]:70s}  [{ty[:28]}]")


# =============================================================================================== (2) refs
BACK = r"(บอก|สอน|เรียก|ใช้|พบ|เห็น|คำนวณ|นิยาม|ตั้ง|ทิ้ง|อธิบาย|แสดง|เขียน|ได้|ให้|เตือน|ชี้|เดิน|เคย|มี|พูด|สัญญา|ยืม|ทำ|ตรวจ|วัด|แยก|ถาม|ตอบ|จบ|เริ่ม|รู้|นับ|อ่าน)"
FWD = r"(จะ|รับช่วง|ถัดไป)"


def sentence_around(text_line, pos, width=240):
    a = max(0, pos - width // 2)
    return re.sub(r"\s+", " ", text_line[a:pos + width]).strip()


def refs(out_dir=None):
    fs = files()
    nsec = {}                                             # chapter -> number of numbered sections, figures, exercises
    for n, f, text in fs:
        body = text.split(ANSWER_MARK)[0]
        secs = [h for h in re.findall(r"^### +(.*)$", body, re.M) if not h.startswith("แบบฝึกหัด")]
        ex = body.split("### แบบฝึกหัด")[-1] if "### แบบฝึกหัด" in body else ""
        nsec[n] = (len(secs), len(re.findall(r"^!\[.*\]\(fig/", body, re.M)), len(re.findall(r"^\d+\. ", ex, re.M)))
    print("=" * 100, "\n(2) sentences that name another chapter\n" + "=" * 100)
    per_part = collections.defaultdict(list)
    mech = []
    total = collections.Counter()
    for n, f, text in fs:
        for i, line, fence, ans in lines_of(text):
            if fence:
                continue
            for m in re.finditer(r"บทที่ (\d+)((?:(?:, | และ |–| กับ | หรือ | ถึง )\d+)*)", line):
                cited = [int(m.group(1))] + [int(x) for x in re.findall(r"\d+", m.group(2))]
                tail = line[m.end():m.end() + 40]
                # "กฎของบทที่ 26 จะ…" is about the rule, not a promise that chapter 26 will do something
                fwd = bool(re.match(r"\s*" + FWD, tail)) and not line[:m.start()].rstrip().endswith("ของ")
                for c in cited:
                    if c == n:
                        continue
                    kind = "FWD" if c > n else "BACK"
                    total[kind] += 1
                    per_part[part_of(n)].append((f.name, i, n, c, kind, fwd, sentence_around(line, m.start())))
                    if c > 30 or c < 1:
                        mech.append(f"  {f.name}:{i}  names chapter {c}, which does not exist")
                    if fwd and c < n and n:
                        mech.append(f"  {f.name}:{i}  'บทที่ {c} จะ…' but chapter {c} is BEFORE chapter {n}: "
                                    f"{sentence_around(line, m.start(), 120)}")
                    if c > n and n and not ans and re.match(r"\s*(บอกว่า|สอนว่า|บอกไว้|สอนแล้ว|สอนไว้|แสดงแล้ว|ใช้แล้ว)", tail):
                        mech.append(f"  {f.name}:{i}  'บทที่ {c} {tail.strip()[:14]}…' in the past tense, but chapter {c} "
                                    f"is AFTER chapter {n}")
            # section N.k, figure N.k, exercise k of chapter N
            for m in re.finditer(r"หัวข้อ(?:ที่)? ?(\d+)\.(\d+)", line):
                c, k = int(m.group(1)), int(m.group(2))
                if c in nsec and not 1 <= k <= nsec[c][0]:
                    mech.append(f"  {f.name}:{i}  หัวข้อ {c}.{k} — chapter {c} has {nsec[c][0]} numbered sections")
            for m in re.finditer(r"รูปที่ (\d+)\.(\d+)", line):
                c, k = int(m.group(1)), int(m.group(2))
                if c in nsec and not 1 <= k <= nsec[c][1]:
                    mech.append(f"  {f.name}:{i}  รูปที่ {c}.{k} — chapter {c} has {nsec[c][1]} figures")
            for m in re.finditer(r"แบบฝึกหัดข้อ (\d+)(?:[^ก-๙]{0,12}| ของ)บทที่ (\d+)|บทที่ (\d+) แบบฝึกหัดข้อ (\d+)", line):
                k, c = (int(m.group(1)), int(m.group(2))) if m.group(1) else (int(m.group(4)), int(m.group(3)))
                if c in nsec and not 1 <= k <= nsec[c][2]:
                    mech.append(f"  {f.name}:{i}  แบบฝึกหัดข้อ {k} ของบทที่ {c} — chapter {c} has {nsec[c][2]} exercises")
    print(f"  {sum(total.values())} chapter-to-chapter references: {total['BACK']} point back, {total['FWD']} point forward")
    for p in sorted(per_part):
        print(f"    part {p or 'preface'}: {len(per_part[p])}")
    print("\n  mechanical defects (existence and direction):")
    print("\n".join(mech) if mech else "    none")
    print("\n  sections / figures / exercises per chapter:", {n: v for n, v in nsec.items() if n})
    if out_dir:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p, items in per_part.items():
            w = [f"# Cross-chapter references made by part {p or 'preface'} ({len(items)})",
                 "# file:line | citing chapter -> cited chapter | direction | sentence", ""]
            w += [f"{fn}:{i} | {n} -> {c} | {kind} | {s}" for fn, i, n, c, kind, fwd, s in items]
            (out_dir / f"refs_part{p}.txt").write_text("\n".join(w) + "\n", encoding="utf-8")
        print(f"\n  wrote {len(per_part)} lists to {out_dir}/refs_partN.txt")


# =============================================================================================== (4) numbers
WATCH = ["187,272", "187,280", "282,625", "125,699", "469,905", "595,604", "312,971", "49,152", "12,288", "16.81",
         "136.5", "0.7975", "0.3058", "1.0356", "1.0779", "1.1847", "3.9544", "0.688", "0.054", "22.5", "124", "103",
         "2,777", "0.8793", "1.0001", "0.6599", "0.6214", "1.3771", "0.2958", "0.3013", "0.1140", "8.7", "41.67", "833"]
NUM = re.compile(r"(?<![\w.,/:])[-+−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.,/:])[-+−]?\d+\.\d{3,}(?![\w])")


def norm(s):
    return s.replace("−", "-").replace("+", "").replace(",", "")


def numbers():
    fs = files()
    key = subprocess.run([sys.executable, str(REPO / "scripts/workbook_answer_key.py")], capture_output=True, text=True).stdout
    key_flat = key.replace(",", "").replace("_", "")
    key_nums = set(re.findall(r"\d+\.\d+|\d+", key_flat))

    def in_key(tok):
        v = norm(tok).lstrip("-")
        if v in key_nums:
            return True
        if "." in v:                                    # the key often prints more digits than the chapter quotes
            d = len(v.split(".")[1])
            x = float(v)
            for k in key_nums:
                if "." in k and len(k.split(".")[1]) > d and k.split(".")[0] == v.split(".")[0]:
                    if abs(round(float(k), d) - x) < 10 ** -(d + 2):
                        return True
        return False

    where = collections.defaultdict(lambda: collections.defaultdict(list))    # value -> chapter -> [lines]
    for n, f, text in fs:
        for i, line, fence, ans in lines_of(text):
            scrub = re.sub(r"`[^`]*\.(?:py|yaml|md|json|pt|svg)[^`]*`", " ", line)        # file.py:123 is not a number
            for m in NUM.finditer(scrub):
                where[norm(m.group(0)).lstrip("-")][n].append(i)
    name = {n: f.name for n, f, _ in fs}

    print("=" * 100, "\n(4a) the watch list of work group 8 (task 4, _HISTORY.md): where each number appears\n" + "=" * 100)
    for w in WATCH:
        hit = collections.defaultdict(list)
        pat = re.compile(r"(?<![\d.,])" + re.escape(w) + r"(?![\d]|,\d|\.\d)")
        for n, f, text in fs:
            for i, line, fence, ans in lines_of(text):
                if pat.search(re.sub(r"`[^`]*\.(?:py|yaml|md|json)[^`]*`", " ", line)):
                    hit[n].append(i)
        ok = in_key(w)
        print(f"  {w:>9s}  key:{'yes' if ok else 'NO '}  chapters {{{', '.join(f'{n}:{len(v)}' for n, v in sorted(hit.items()))}}}")

    print("\n" + "=" * 100, "\n(4b) numbers that appear in two or more chapters and are NOT printed by the answer key\n" + "=" * 100)
    for v, chs in sorted(where.items(), key=lambda kv: -len(kv[1])):
        if len(chs) >= 2 and not in_key(v):
            print(f"  {v:>12s}  " + "  ".join(f"{name[n]}:{ls[0]}" + (f"(+{len(ls) - 1})" if len(ls) > 1 else "") for n, ls in sorted(chs.items())))

    print("\n" + "=" * 100, "\n(4c) near-twins: the same quantity quoted with a different last digit in different chapters\n" + "=" * 100)
    print("Two decimals with the same number of places that differ by exactly one unit in the last place, each in a\n"
          "different chapter. Most pairs are different quantities that happen to be close; a real twin is a rounding slip.\n")
    decs = sorted((v for v in where if "." in v), key=float)
    by_places = collections.defaultdict(list)
    for v in decs:
        by_places[len(v.split(".")[1])].append(v)
    for d, vs in by_places.items():
        if d < 4:
            continue
        for a, b in zip(vs, vs[1:]):
            if abs(float(b) - float(a) - 10 ** -d) < 10 ** -(d + 3) and set(where[a]) != set(where[b]):
                print(f"  {a} {dict((name[n][:2], ls[:3]) for n, ls in where[a].items())}   vs   "
                      f"{b} {dict((name[n][:2], ls[:3]) for n, ls in where[b].items())}")

    print("\n" + "=" * 100, "\n(4d) distinctive numbers in ONE chapter only that the answer key does not print\n" + "=" * 100)
    miss = collections.defaultdict(list)
    for v, chs in where.items():
        if len(chs) == 1 and not in_key(v):
            (n, ls), = chs.items()
            miss[n].append(f"{v}@{ls[0]}")
    for n in sorted(miss):
        print(f"  {name[n]}: {len(miss[n])}  " + "  ".join(miss[n][:14]) + (" …" if len(miss[n]) > 14 else ""))


# =============================================================================================== (5) conventions
# (label, regex over prose, chapters where the pattern must NOT appear, the agreement it comes from)
RULES = [
    ("รอบ = step only in ch 3–6", r"(?<![ก-๙])(?:ทุก|แต่ละ|หนึ่ง|ต่อ)รอบ(?!กวาด|ของ policy|การ)", range(7, 31), "ส่วนที่ 2: ตั้งแต่บทที่ 7 ใช้ step"),
    ("state in Latin", r"\bstates?\b", range(1, 31), "ส่วนที่ 2: ใช้รูปไทยว่าสถานะ"),
    ("โครงข่ายของ policy after ch 20", r"โครงข่ายของ policy", range(21, 31), "ส่วนที่ 3 / บทที่ 16–21: ตั้งแต่บทที่ 20 ใช้ actor"),
    ("actor before ch 20", r"\bactor\b", range(1, 20), "ส่วนที่ 3: ① เรียกว่าโครงข่ายของ policy จนถึงบทที่ 20"),
    ("โครงข่ายตัวที่สาม after ch 21", r"โครงข่ายตัวที่สาม", range(22, 31), "บทที่ 16–21: ตั้งแต่บทที่ 21 ใช้ encoder"),
    ("encoder before ch 21", r"\bencoder\b", range(1, 21), "บทที่ 16–21"),
    ("mini-batch before ch 27", r"\bmini-?batch", range(1, 27), "บทที่ 23–24: ไม่ใช้คำว่า mini-batch"),
    ("epoch before ch 27", r"\bepochs?\b", range(1, 27), "บทที่ 23–24: ไม่ใช้คำว่า epoch"),
    ("KL before ch 26", r"\bKL\b", range(1, 26), "บทที่ 23–24: ไม่ใช้คำว่า KL"),
    ("TRPO anywhere before ch 26", r"\bTRPO\b", range(1, 26), "บทที่ 23–24: ไม่เอ่ยถึง TRPO"),
    ("ชุดของช่อง / ชุดที่ N after ch 27", r"ชุดของช่อง|ชุดที่ [12]", range(27, 31), "บทที่ 25–27: ตั้งแต่บทที่ 27 ใช้ mini-batch"),
    ("variance / bias (statistical) outside ch 17", r"\bvariance\b", [n for n in range(1, 31) if n != 17], "บทที่ 16–21: ปรากฏครั้งเดียวในหมายเหตุของบทที่ 17"),
    ("หนีบ used for clip_grad_norm_", r"clip_grad_norm_[^.]{0,60}หนีบ|หนีบ[^.]{0,40}clip_grad_norm_", range(1, 31), "บทที่ 25–27: กริยาของ clip_grad_norm_ คือย่อ"),
    ("ช่วง A–D of the iteration", r"ช่วง [A-D](?![a-zA-Z])", range(1, 31), "บทที่ 25–27: สี่ช่วงเรียก ก ข ค ง"),
    ("ตอน/ช่วง A–D of the value line", r"ตอน(?:ที่)? ?[A-D](?![a-zA-Z])", range(1, 31), "บทที่ 28–30: ตอนที่ 1–4"),
    ("โปรแกรมบนหุ่น", r"โปรแกรมบนหุ่น|รันบนหุ่น|ทำงานบนหุ่น|อยู่บนหุ่น|ไปอยู่บนหุ่น|บนตัวหุ่นจริง", range(1, 31), "บทที่ 28–30: โปรแกรมของชั้น deploy / ไปถึงชั้น deploy"),
    ("รุ่น for software", r"(?:PyTorch|Python|Isaac ?(?:Sim|Lab)|ROS ?2?|limxsdk|ONNX|numpy|conda|Ubuntu|CUDA)[^.|]{0,25}รุ่น|รุ่น ?\d+\.\d+", range(0, 31), "_STYLE.md §4: รุ่น = PF/SF/WF, ซอฟต์แวร์ใช้เวอร์ชัน"),
    ("exclamation mark", r"(?<!\[)(?<!<)!(?!\[|=|--)", range(0, 31), "_STYLE.md §3: ไม่มีเครื่องหมายอัศเจรีย์"),
    ("ภาค", r"ภาค(?!ผนวก)", range(0, 31), "task 7: ใช้ส่วนที่"),
    ("Thai digit", r"[๐-๙]", range(0, 31), "task 8: เลขอารบิกทั้งเล่ม"),
    ("shape of a tensor without the space", r"", [], ""),          # handled below (code spans)
    ("epsilon-greedy in Greek", r"[εϵ]-greedy", range(0, 31), "ส่วนที่ 3: เขียนด้วยอักษรละติน"),
    ("ข้อมูลเก่า / policy เวอร์ชันก่อน inside one table", r"policy (?:เวอร์ชัน|รุ่น)ก่อน", range(0, 31), "ข้อบกพร่องของ 05 §4.7 ที่แก้แล้ว"),
    ("clip as a cap on ρ", r"(?:ไม่เกิน|ไม่ให้เกิน|จำกัด)[^.|]{0,30}(?:20 ?%|ร้อยละ 20)", range(0, 31), "ข้อบกพร่องที่แก้แล้ว: clip ไม่ใช่เพดานของ ρ"),
]
# (label, regex over a math chunk, chapters where it must not appear, source)
MATH_RULES = [
    ("bare A, no hat and no subscript (glossary §7)", r"(?<![a-zA-Z\\_^{}])(?<!hat )(?<!hat\{)(?<!bar )(?<!bar\{)A(?![a-zA-Z_^({\\}])\s*(?:$|[=,+)\-])", range(0, 31), "อภิธานศัพท์ §7: ห้ามเขียน A ลอย ๆ (\\bar A และ sd_A ของสมการ (28) เป็นสัญลักษณ์ที่มีนิยาม)"),
    ("bare R (glossary §7)", r"(?<![a-zA-Z\\_^{])R(?![a-zA-Z_^({\\])\s*(?:$|[=,+)\-])", range(0, 31), "อภิธานศัพท์ §7: ห้ามเขียน R ลอย ๆ"),
    ("\\epsilon before ch 12 / as noise", r"\\epsilon\s*\\sim|\\epsilon_t|\\mu\s*\+\s*\\sigma\s*\\epsilon", range(0, 31), "อภิธานศัพท์ §7: noise คือ \\varepsilon"),
    ("\\varepsilon as the clip range", r"1\s*[-+]\s*\\varepsilon|\\varepsilon\s*=\s*0\.2", range(0, 31), "อภิธานศัพท์ §7: ขอบ clip คือ \\epsilon"),
    ("\\hat v without k in an Adam formula", r"\\sqrt\{\\hat\{? ?v\}?\}(?!_)|\\hat\{? ?v\}?\s*\}?\s*\+\s*\\mathrm\{eps", range(0, 31), "บทที่ 25–27: \\hat v_k มีตัวห้อย k เสมอ"),
    ("uppercase random variable X, K outside ch 4", r"(?<![a-zA-Z\\_^{])[XK](?![a-zA-Z])", [n for n in range(0, 31) if n != 4], "ส่วนที่ 2: เฉพาะบทที่ 4"),
    ("variance squared N(mu, sigma^2)", r"\\mathcal\{?N\}?\s*\([^)]*\\sigma\^2", range(0, 31), "อภิธานศัพท์ §7: 𝒩(μ, σ) โดย σ คือส่วนเบี่ยงเบนมาตรฐาน"),
]


def conventions():
    fs = files()
    print("=" * 100, "\n(5) word agreements, (7) ภาค, (8) Thai digits\n" + "=" * 100)
    for label, pat, where, src in RULES:
        if not pat:
            continue
        hits = []
        for n, f, text in fs:
            if n not in where:
                continue
            for i, line, fence, ans in lines_of(text):
                if fence:
                    continue
                prose, _, _ = split_kinds(line)
                for m in re.finditer(pat, prose):
                    hits.append(f"      {f.name}:{i}  …{line[max(0, m.start() - 45):m.end() + 45].strip()}…")
        print(f"\n  {label}  [{src}]: {len(hits)}")
        print("\n".join(hits[:25]) + (f"\n      … {len(hits) - 25} more" if len(hits) > 25 else "") if hits else "      none")
    for label, pat, where, src in MATH_RULES:
        hits = []
        for n, f, text in fs:
            if n not in where:
                continue
            for i, line, fence, ans in lines_of(text):
                if fence:
                    continue
                for m in split_kinds(line)[1]:
                    inner = m.strip("$")
                    if re.search(pat, inner):
                        hits.append(f"      {f.name}:{i}  {m[:90]}")
        print(f"\n  {label}  [{src}]: {len(hits)}")
        print("\n".join(hits[:25]) + (f"\n      … {len(hits) - 25} more" if len(hits) > 25 else "") if hits else "      none")
    # tensor shapes are written `(B, 8)` with a space after the comma
    hits = []
    for n, f, text in fs:
        for i, line, fence, ans in lines_of(text):
            for c in split_kinds(line)[2]:
                if re.fullmatch(r"`\((?:[A-Za-z0-9]+,)+[A-Za-z0-9]+\)`", c):
                    hits.append(f"      {f.name}:{i}  {c}")
    print(f"\n  tensor shape without a space after the comma  [บทที่ 28–30]: {len(hits)}")
    print("\n".join(hits[:25]) if hits else "      none")


# =============================================================================================== (6) dots
def dots(pdf=PDF, dpi=150, show=True):
    """A page whose LAST band of ink is too short and too narrow to be a line of text.

    When a heading starts a page, Chrome paints the part of its glyphs that sticks out above the line box (the upper
    tone mark of a stacked Thai cluster) on the page BEFORE, as a speck at the very bottom of the text block. A real
    last line is at least 2 mm tall; a rule or a table border is thin but as wide as the block. A speck is neither.
    """
    import tempfile
    try:
        from PIL import Image
    except ImportError:
        sys.exit("dots needs Pillow")
    if show:
        print("=" * 100, f"\n(6) specks at the bottom of the text block of {pdf.name}\n" + "=" * 100)
    mm = dpi / 25.4
    bad = []
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", str(pdf), d + "/p"], check=True)
        pages = sorted(pathlib.Path(d).glob("p-*.png"))
        for p in pages[1:]:                              # page 1 is the cover: absolutely placed, no text block
            im = Image.open(p)
            w, h = im.size
            # the text block: 19 mm side margins, 24 mm top, 22 mm bottom (CSS @page of workbook_to_pdf.py)
            block = im.crop((int(17 * mm), int(24 * mm), int(w - 17 * mm), int(h - 21 * mm))).point(lambda v: 255 if v < 140 else 0)
            bw, bh = block.size
            rows = [block.crop((0, y, bw, y + 1)).getbbox() for y in range(bh)]
            y = bh - 1
            while y >= 0 and rows[y] is None:
                y -= 1
            if y < 0:
                continue
            bottom, left, right = y, bw, 0
            while y >= 0 and rows[y] is not None:
                left, right = min(left, rows[y][0]), max(right, rows[y][2])
                y -= 1
            height_mm, width_mm = (bottom - y) / mm, (right - left) / mm
            gap_mm = 0
            while y >= 0 and rows[y] is None:
                y -= 1
                gap_mm += 1 / mm
            if height_mm < 1.3 and width_mm < 60 and gap_mm > 1.0:
                bad.append((int(p.stem.split("-")[1]), round(height_mm, 2), round(width_mm, 1), round(gap_mm, 1)))
    if show:
        print(f"  {len(pages)} pages, {len(bad)} whose last ink is a speck")
        for n, hh, ww, gg in bad:
            print(f"    page {n}: last ink is {hh} mm tall, {ww} mm wide, {gg} mm below the line above it")
    return bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    what = args[0] if args else "all"
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if what in ("terms", "all"):
        terms()
    if what in ("refs", "all"):
        refs(out)
    if what in ("numbers", "all"):
        numbers()
    if what in ("conventions", "all"):
        conventions()
    if what in ("dots",):
        dots()


if __name__ == "__main__":
    main()
