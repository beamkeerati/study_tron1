#!/usr/bin/env python3
"""Build notes/workbook/NN-*.md into one textbook-style PDF: notes/pdf/08-workbook.pdf

Book design ported from KMU_Korean/scripts/make_pdf.py (cover, TOC with page numbers, running
head per chapter, page x / y, callout boxes, keep-together paging rules, Laksaman type).
The renderer is Chrome, not WeasyPrint: WeasyPrint cannot run MathJax, and feeding it MathJax's
SVG drops tagged/boxed displays and collapses aligned ones (tested 2026-11-25). Chrome also
breaks Thai lines natively, so KMU's U+200B pass is not needed here.

What Chrome lacks, and how this script supplies it:
  - string-set running heads  -> one named @page per chapter (`page: chN`) with a literal title
  - target-counter TOC pages  -> two passes: pass 1 prints invisible ASCII markers, pdftotext
                                 finds each marker's page, pass 2 prints the numbers

Content conventions (see notes/workbook/_STYLE.md):
  The table of contents is clickable (each row links to its part page or chapter title) and the PDF carries
  bookmarks built by Chrome from the headings.
  `# ...`  part title page (its own file NN-0-part.md, NN = first chapter of the part)
  `## บทที่ N · ชื่อ — คำขยาย`  chapter (new page, running head); a `## ` without บทที่ N is an unnumbered
  chapter (00-preface.md): no section numbers, no INTRO/SUMMARY check
  `### ...` numbered section N.k (a heading starting with แบบฝึกหัด is not numbered)
  > [!INTRO] > [!SUMMARY] > [!NOTE] > [!WARNING] > [!REMARK] > [!KEY] ชื่อสมการ
  > [!EXAMPLE] โจทย์   (body = วิธีทำ; the closing # is added)
  <!--space:NN-->  NN mm of blank working space after an exercise item
  ![caption](fig/name.svg|png)  on its own line: a figure, numbered N.k (SVG is inlined so it uses the page font)
  <!-- เฉลย -->    everything after this line in a chapter file moves to the answer key

Checks printed after every build: mjx-merror (must be 0), elements wider than the text block,
chapters missing INTRO/SUMMARY, and specks at the foot of a page (a Thai tone mark that Chrome painted on the page
before its heading; scripts/workbook_audit.py `dots`).

Usage (from anywhere):  python3 scripts/workbook_to_pdf.py [--keep-html]
Needs: pip install --user markdown ; google-chrome ; poppler-utils ; fonts-tlwg-laksaman, fonts-noto.
"""
import html
import pathlib
import re
import subprocess
import sys

import markdown

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / 'notes/workbook'
OUT = REPO / 'notes/pdf'
NAME = '08-workbook'
MATHJAX = (REPO / 'notes/diagrams/mathjax/tex-svg.js').as_uri()
TLWG = 'file:///usr/share/fonts/truetype/tlwg'
ANSWER_MARK = '<!-- เฉลย -->'
TEXT_W_MM = 172                       # A4 210 mm minus 19 mm side margins

# ---------------------------------------------------------------- protect math / code
def protect(src, store):
    def keep(m):
        store.append(re.sub(r'\n> ?', '\n', m.group(0)))      # a display inside a `> ` callout: drop the quote prefixes
        return f'MATHPH{len(store) - 1:05d}ZZ'
    src = re.sub(r'```.*?```', keep, src, flags=re.S)                    # fenced code
    src = re.sub(r'\$\$.*?\$\$', keep, src, flags=re.S)                  # display math
    src = re.sub(r'(?<!\\)\$(?!\s)(?:\\.|[^$\\])+?\$', keep, src)        # inline math
    return src


def thai_in_math(store):
    return [s for s in store if not s.startswith('```') and re.search(r'[\u0e00-\u0e7f]', s)]


def restore(h, store):
    def back(m):
        s = store[int(m.group(1))]
        if s.startswith('```'):
            body = s.split('\n', 1)[1].rsplit('```', 1)[0]
            return f'<pre><code>{html.escape(body)}</code></pre>'
        if s.startswith('$$'):
            return r'\[' + html.escape(s[2:-2]) + r'\]'
        return r'\(' + html.escape(s[1:-1]) + r'\)'
    h = re.sub(r'<p>(MATHPH\d{5}ZZ)</p>', lambda m: m.group(1) if store[int(m.group(1)[6:11])].startswith('```') else m.group(0), h)
    return re.sub(r'MATHPH(\d{5})ZZ', back, h)


MD = markdown.Markdown(extensions=['tables', 'fenced_code', 'sane_lists'])


def render_md(text):
    MD.reset()
    return MD.convert(text)


# ---------------------------------------------------------------- callouts
LABELS = {'intro': 'บทนี้ว่าด้วยอะไร', 'summary': 'สรุปท้ายบท', 'note': 'หมายเหตุ', 'warning': 'ระวัง',
          'remark': 'Remark', 'tip': 'เคล็ดลับ', 'key': '', 'example': 'ตัวอย่าง'}
CALLOUT = re.compile(r'^> ?\[!(INTRO|SUMMARY|NOTE|WARNING|REMARK|TIP|KEY|EXAMPLE)\][ \t]*([^\n]*)\n((?:>.*\n?)*)', re.M)


def convert_callouts(text):
    def repl(m):
        kind, title = m.group(1).lower(), m.group(2).strip()
        body = '\n'.join(re.sub(r'^> ?', '', line) for line in m.group(3).splitlines())
        inner = render_md(body)
        if kind == 'example':
            q = render_md(title)
            q = re.sub(r'^<p>', '<p class="exq"><span class="lbl">ตัวอย่าง</span> ', q, count=1)
            return (f'<div class="ex">{q}<div class="exa"><span class="lbl">วิธีทำ</span>{inner}</div>'
                    f'<div class="qed">#</div></div>\n\n')
        if kind == 'remark':
            return f'<div class="box box-remark"><span class="rbadge">R</span><div class="rbody">{inner}</div></div>\n\n'
        if kind == 'note':                      # Math Center's inline "หมายเหตุ:" — a label, not a box
            inner = re.sub(r'^<p>', f'<p><span class="lbl-note">{title or LABELS[kind]}:</span> ', inner, count=1)
            return f'<div class="note">{inner}</div>\n\n'
        t = title or LABELS[kind]
        t = re.sub(r'^<p>|</p>$', '', render_md(t)) if t else ''
        return f'<div class="box box-{kind}"><div class="box-title">{t}</div>{inner}</div>\n\n'
    return CALLOUT.sub(repl, text)


# ---------------------------------------------------------------- CSS
CSS = r"""
@font-face { font-family: 'WB Thai'; font-weight: normal; font-style: normal; src: url('__TLWG__/Laksaman.ttf'); }
@font-face { font-family: 'WB Thai'; font-weight: bold;   font-style: normal; src: url('__TLWG__/Laksaman-Bold.ttf'); }
@font-face { font-family: 'WB Thai'; font-weight: normal; font-style: italic; src: url('__TLWG__/Laksaman-Italic.ttf'); }
@font-face { font-family: 'WB Thai'; font-weight: bold;   font-style: italic; src: url('__TLWG__/Laksaman-BoldItalic.ttf'); }
:root { --ink: #1f2328; --accent: #0f6e6e; --rule: #c9d1d9; --muted: #6e7781; }
@page { size: A4; margin: 24mm 19mm 22mm 19mm;
  @top-right    { content: "__BOOK__"; font-family: 'WB Thai'; font-size: 10.5px; color: #6e7781; vertical-align: bottom; padding-bottom: 6px; }
  @bottom-right { content: counter(page) " / " counter(pages); font-family: 'WB Thai'; font-size: 10.5px; color: #6e7781; vertical-align: top; padding-top: 6px; }
}
@page :first { @top-right { content: none; } @bottom-right { content: none; } @top-left { content: none; } }
__PAGES__
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'WB Thai', 'Noto Sans Thai', sans-serif; font-size: 14.5px; line-height: 1.66; color: var(--ink); margin: 0; }
body.check { width: __W__mm; }
p { margin: 0 0 0.7em; orphans: 3; widows: 3; }
strong { color: #0b2e4f; }
a { color: inherit; text-decoration: none; }
code { font-family: 'Noto Sans Mono', monospace; font-size: 0.78em; background: #f3f5f7; padding: 1px 4px; border-radius: 3px; }
pre { font-family: 'Noto Sans Mono', monospace; font-size: 0.78em; background: #f6f8fa; padding: 9px 12px; border: 1px solid #d0d7de;
      border-radius: 5px; white-space: pre-wrap; line-height: 1.45; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 1em; }
hr { border: none; border-top: 1px solid var(--rule); margin: 1.4em 0; }

/* ---- cover (Math Center: the title sits between two rules) and TOC ---- */
.cover { height: 250mm; position: relative; break-after: page; }
.cover .ctitle { position: absolute; top: 38mm; right: 0; width: 118mm; border-top: 5px solid var(--accent); border-bottom: 5px solid var(--accent);
                 padding: 16px 0 18px; }
.cover h1 { font-size: 34px; line-height: 1.3; margin: 0 0 6px; color: var(--ink); }
.cover .ctitle p { margin: 0; color: #4b5563; font-size: 1.05em; }
.cover .cfoot { position: absolute; bottom: 0; right: 0; text-align: right; color: var(--muted); font-size: 0.92em; }
.toc { break-after: page; }
.toc h2 { font-size: 22px; color: var(--accent); margin: 0 0 14px; }
.toc a { display: flex; align-items: baseline; margin: 3px 0; color: inherit; text-decoration: none; }
.toc a.part { font-weight: bold; margin-top: 12px; }
.toc .dots { flex: 1; border-bottom: 1px dotted #8c959f; margin: 0 6px; transform: translateY(-4px); }
.toc .pg { color: var(--muted); font-variant-numeric: tabular-nums; min-width: 2.2em; text-align: right; }

/* ---- headings: part page, chapter pill (Math Center), numbered sections (Becker) ---- */
h1.part { font-size: 30px; color: var(--accent); margin: 90mm 0 0; padding-bottom: 14px; border-bottom: 3px solid var(--accent); break-before: page; break-after: page; }
section.chapter { break-before: page; }
.chno { font-size: 13px; color: var(--muted); letter-spacing: 0.04em; margin: 0 0 4px; }
h2.chapter { display: inline-block; position: relative; font-size: 21px; line-height: 1.4; font-weight: bold; margin: 0 0 18px; padding: 7px 16px 6px;
             background: #eceff1; border: 1.3px solid #37474f; border-radius: 8px; break-after: avoid; }
.pm { position: absolute; left: 0; top: 0; font-size: 1px; line-height: 1px; color: transparent; }
/* Space above a heading is padding, not margin. At the top of a page Chrome drops the margin, and the upper tone mark of a
   stacked Thai cluster, which rises above the line box, was then painted at the foot of the page before (12 pages of the
   2026-11-26 build). Padding stays with the box, so the mark stays on its own page. */
h3 { font-size: 17.5px; line-height: 1.4; margin: 16px 0 8px; padding-top: 12px; break-after: avoid; }
h3 .secno { color: var(--accent); margin-right: 10px; }
h3.exh { text-decoration: underline; text-underline-offset: 4px; font-size: 16px; margin-top: 18px; }
h4 { font-size: 15px; margin: 8px 0 6px; padding-top: 10px; color: #1a4d7a; break-after: avoid; }

/* ---- tables: open, compact; a short table is one unit ---- */
table { border-collapse: collapse; width: 100%; margin: 0.2em 0 1.2em; font-size: 0.93em; line-height: 1.5;
        border-top: 1.6px solid #37474f; border-bottom: 1.2px solid #78868f; }
th, td { padding: 5px 10px; vertical-align: top; text-align: left; border-bottom: 1px solid #e4e9ef; }
th { border-bottom: 1.2px solid #37474f; font-size: 0.94em; color: #37474f; }
tbody tr:last-child td { border-bottom: none; }
tr { break-inside: avoid; }
table.keep { break-inside: avoid; }
tbody > tr:last-child { break-before: avoid; }
thead { display: table-header-group; }
p.lead { break-after: avoid; margin-bottom: 0.35em; }
td mjx-container, th mjx-container { margin: 0 !important; }

/* ---- lists ---- */
ul, ol { margin: 0.1em 0 1em; padding-left: 1.8em; }
li { margin: 0 0 0.3em; }
li > ul, li > ol { margin: 0.25em 0 0.3em; }

/* ---- boxes ---- */
.box { border-left: 5px solid; border-radius: 0 7px 7px 0; padding: 10px 16px; margin: 0.7em 0 1.3em; font-size: 0.96em; line-height: 1.6; break-inside: avoid; }
.box > :last-child, .rbody > :last-child, .note > :last-child, .exa > :last-child { margin-bottom: 0; }
.box-title { font-weight: bold; margin-bottom: 4px; break-after: avoid; }
.box-intro   { background: #f0f6fc; border-color: #0f4c81; break-inside: auto; }
.box-intro .box-title { color: #0f4c81; }
.box-summary { background: #f6f8fa; border-color: #57606a; }
.box-warning { background: #fff8c5; border-color: #9a6700; }
.box-tip     { background: #dafbe1; border-color: #1a7f37; }
/* Becker: a named equation in a tinted frame */
.box-key { background: #f1f8f8; border: 1px solid var(--accent); border-radius: 0; padding: 9px 16px 6px; }
.box-key .box-title { color: var(--accent); font-size: 1.04em; }
.box-key table { margin: 0.2em 0 0.4em; border: none; font-size: 0.95em; }
.box-key td, .box-key th { border: none; padding: 1px 10px 1px 0; }
/* Becker: Remark with the circled R */
.box-remark { border: none; padding: 0 0 0 40px; position: relative; font-size: 0.93em; color: #333; margin: 0.6em 0 1.1em; }
.rbadge { position: absolute; left: 6px; top: 2px; width: 22px; height: 22px; border-radius: 50%; background: #cfe9e9; border: 1px solid var(--accent);
          color: var(--accent); font-weight: bold; font-size: 12px; line-height: 22px; text-align: center; }
/* Math Center: หมายเหตุ / ตัวอย่าง / วิธีทำ / # */
.note { margin: 0.5em 0 1em; padding-left: 0; }
.lbl-note { font-weight: bold; margin-right: 6px; }
.ex { margin: 1.1em 0 1.3em; }
.ex .lbl { text-decoration: underline; text-underline-offset: 3px; font-weight: bold; margin-right: 8px; }
.exq { margin-bottom: 0.45em; break-after: avoid; }
.exa { padding-left: 0; }
.exa > .lbl { float: left; }
.exa > p:first-of-type { margin-top: 0; }
.qed { text-align: right; font-weight: bold; margin-top: -0.6em; }
blockquote { margin: 0.6em 0 1.2em; padding: 8px 16px; border-left: 4px solid #94a3b8; background: #f8fafc; }

/* ---- exercises: working space under each item, sub-items in two columns ---- */
.exercises > ol > li { margin-bottom: 9mm; break-inside: avoid; }
.exercises > ol > li > ol { columns: 2; column-gap: 26px; margin-top: 0.5em; }
.exercises > ol > li > ol > li { break-inside: avoid; margin-bottom: 0.9em; }
.ws { display: block; }
.src { color: var(--muted); font-size: 0.9em; white-space: nowrap; }
.answers h3 { font-size: 15.5px; margin: 14px 0 8px; display: table; padding: 3px 12px; background: #eceff1; border: 1px solid #37474f; border-radius: 6px; }
.answers { font-size: 0.95em; }
.answers ol { margin-bottom: 0.6em; }
.answers li > ol { columns: 2; column-gap: 26px; }
.answers li > ol > li { break-inside: avoid; }

mjx-container[display="true"] { margin: 8px 0 !important; break-inside: avoid; }
/* figures: a new inline SVG (preferred) or a crop from a textbook; numbered per chapter by the builder */
figure { margin: 1.1em 0 1.4em; text-align: center; break-inside: avoid; }
figure svg, figure img { max-width: 100%; height: auto; }
figcaption { font-size: 0.92em; color: #4b5563; margin-top: 6px; text-align: center; }
figcaption b { color: var(--ink); margin-right: 6px; }
/* MathJax gives every glyph stroke-width 0; in a PDF a zero-width stroke is drawn as the thinnest line the device has,
   which makes all math look bold on screen. Glyphs are filled paths, so drop their stroke (frames from \boxed keep theirs). */
mjx-container svg path, mjx-container svg use { stroke: none; }
/* Math Center: the working of an example starts at the left, under its label */
.exa mjx-container[display="true"] { text-align: left !important; margin-left: 3.4em !important; }
/* Becker's word-box reading of an equation; Thai never goes inside math (MathJax places Thai marks wrongly) */
.wordeq { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 8px 10px; margin: 0.7em 0 1.2em; break-inside: avoid; }
.wordeq span { border: 1.2px solid #37474f; padding: 3px 10px; background: #fff; }
"""

HEAD = """<!doctype html><html lang="th"><head><meta charset="utf-8"><title>%(title)s</title><style>%(css)s</style>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']],packages:{'[+]':['ams','boldsymbol']}},
svg:{fontCache:'global',mtextInheritFont:true}};</script>
<script src="%(mj)s"></script></head><body class="%(cls)s">"""

# Appended to the check pass only. The vendored MathJax build does not call startup hooks, so wait until the number of
# typeset containers stops changing, then flag every element whose right edge is past the text block.
MEASURE = """<script>(function(){var last=-1,same=0,t=setInterval(function(){
  var n=document.querySelectorAll('mjx-container svg').length; same=(n===last)?same+1:0; last=n;
  if(same<8)return; clearInterval(t);
  var lim=document.body.getBoundingClientRect().right+1,k=0;
  document.querySelectorAll('mjx-container,table,pre,img,figure svg').forEach(function(el){
    if(el.getBoundingClientRect().right>lim){k++;el.setAttribute('data-overflow','1');}});
  document.body.setAttribute('data-overflow-count',k);},100);})();</script>"""

SOURCE_TAG = re.compile(r'\[((?:0[0-7]|อภิธานศัพท์)(?: §[^\]\n]{1,40})?|[a-z_/]+\.py:\d+(?:-\d+)?)\](?!\()')


def plain(s):
    """Heading HTML -> text for the TOC / running head (math and tags dropped)."""
    s = re.sub(r'MATHPH\d{5}ZZ', '', s)
    s = html.unescape(re.sub(r'<[^>]+>', '', s))
    return re.sub(r'\s+', ' ', s).strip(' ·—-')


def css_str(s):
    return s.replace('\\', '\\\\').replace('"', '\\"')


# ---------------------------------------------------------------- assemble
def load():
    files = sorted(SRC.glob('[0-9][0-9]-*.md'))
    if not files:
        sys.exit(f'no chapter files in {SRC}')
    cover, body, answers = '', [], []
    for f in files:
        text = f.read_text(encoding='utf-8')
        if f.name == '00-cover.md':
            cover = text
            continue
        main, _, ans = text.partition(ANSWER_MARK)
        body.append(main.strip() + '\n')
        m = re.search(r'^## +(.+)$', main, re.M)
        if ans.strip() and m:
            answers.append((m.group(1).split('—')[0].strip(), ans.strip()))
        for kind in ('INTRO', 'SUMMARY'):
            if m and m.group(1).startswith('บทที่') and f'[!{kind}]' not in main:
                print(f'  style: {f.name} has no [!{kind}] box')
    return cover, '\n\n'.join(body), answers


def to_html(text, store):
    text = protect(text, store)
    text = re.sub(r'<!--\s*space:(\d+)\s*-->', r'WSPACE\1ZZ', text)
    text = SOURCE_TAG.sub(r'<span class="src">[\1]</span>', text)
    text = re.sub(r'^!\[(.*?)\]\((fig/[^)\s]+)\)[ \t]*$', r'<figure data-src="\2"><figcaption>\1</figcaption></figure>', text, flags=re.M)
    h = render_md(convert_callouts(text))
    h = re.sub(r'WSPACE(\d+)ZZ', r'<span class="ws" style="height:\1mm"></span>', h)
    # a short paragraph right before a table goes with the table; a short table is never split
    h = re.sub(r'<p>((?:(?!</p>).)*)</p>(\s*)<table', lambda m: ('<p class="lead">' if len(re.sub(r'<[^>]+>', '', m.group(1))) < 220
                                                                    else '<p>') + m.group(1) + '</p>' + m.group(2) + '<table', h, flags=re.S)
    h = re.sub(r'<table>(.*?)</table>', lambda m: ('<table class="keep">' if m.group(1).count('<tr') <= 9 else '<table>')
               + m.group(1) + '</table>', h, flags=re.S)
    return h


def structure(h):
    """Split at h1/h2: wrap chapters in <section>, number the h3s, mark exercise blocks. Returns (html, toc entries, @page css)."""
    parts = re.split(r'(?=<h[12]>)', h)
    out, toc, pages, n = [parts[0]], [], [], 0
    for chunk in parts[1:]:
        n += 1
        mark = f'@@c{n:03d}@@'
        if chunk.startswith('<h1>'):
            title = re.match(r'<h1>(.*?)</h1>', chunk, re.S).group(1)
            toc.append((mark, plain(title), 'part'))
            out.append(re.sub(r'<h1>(.*?)</h1>', rf'<h1 class="part" id="c{n:03d}" style="position:relative"><span class="pm">{mark}</span>\1</h1>',
                              chunk, count=1, flags=re.S))
            continue
        title = re.match(r'<h2>(.*?)</h2>', chunk, re.S).group(1)
        m = re.match(r'\s*(บทที่\s*(\d+))\s*·\s*(.*)', title, re.S)
        label, num, rest = (m.group(1), m.group(2), m.group(3)) if m else ('', '', title)
        short = plain(rest.split('—')[0])
        toc.append((mark, (f'{label} · ' if label else '') + plain(rest), 'ch'))
        pages.append(f'@page ch{n} {{ @top-left {{ content: "{css_str((label + " · ") if label else "")}{css_str(short)}"; font-family: \'WB Thai\'; '
                     f'font-size: 10.5px; color: #6e7781; vertical-align: bottom; padding-bottom: 6px; }} }}')
        head = (f'<div class="chno">{label}</div>' if label else '') + f'<h2 class="chapter" id="c{n:03d}"><span class="pm">{mark}</span>{rest}</h2>'
        chunk = re.sub(r'<h2>.*?</h2>', lambda _: head, chunk, count=1, flags=re.S)
        k, is_answers = 0, 'เฉลย' in plain(title)[:12]

        def number(mm):
            nonlocal k
            if plain(mm.group(1)).startswith('แบบฝึกหัด'):
                return f'<h3 class="exh">{mm.group(1)}</h3>'
            if is_answers or not num:
                return mm.group(0)
            k += 1
            return f'<h3><span class="secno">{num + "." if num else ""}{k}</span>{mm.group(1)}</h3>'
        chunk = re.sub(r'<h3>(.*?)</h3>', number, chunk, flags=re.S)
        chunk = re.sub(r'(<h3 class="exh">.*?</h3>)(.*?)(?=<h3|$)', r'\1<div class="exercises">\2</div>', chunk, flags=re.S)
        fno = 0

        def figure(mm):
            nonlocal fno
            fno += 1
            f = SRC / mm.group(1)
            pic = (re.sub(r'<\?xml.*?\?>', '', f.read_text(encoding='utf-8')) if f.suffix == '.svg'
                   else f'<img src="{f.as_uri()}" alt="">')
            return (f'<figure>{pic}<figcaption><b>รูปที่ {num + "." if num else ""}{fno}</b> {mm.group(2)}</figcaption></figure>')
        chunk = re.sub(r'<figure data-src="([^"]+)"><figcaption>(.*?)</figcaption></figure>', figure, chunk, flags=re.S)
        cls = 'chapter answers' if is_answers else 'chapter'
        out.append(f'<section class="{cls}" style="page: ch{n}">{chunk}</section>')
    return ''.join(out), toc, '\n'.join(pages)


def cover_html(text, store):
    h = to_html(text, store)
    m = re.match(r'\s*<h1>(.*?)</h1>(.*)', h, re.S)
    title, rest = m.group(1), m.group(2)
    lead, _, foot = rest.partition('<hr />')
    return f'<div class="cover"><div class="ctitle"><h1>{title}</h1>{lead}</div><div class="cfoot">{foot}</div></div>', plain(title)


def toc_html(toc, page_of):
    rows = ''.join(f'<a class="{kind}" href="#{mark.strip("@")}"><span>{html.escape(t)}</span><span class="dots"></span>'
                   f'<span class="pg">{page_of.get(mark, "")}</span></a>' for mark, t, kind in toc)
    return f'<div class="toc"><h2>สารบัญ</h2>{rows}</div>'


def chrome(*args):
    return subprocess.run(['google-chrome', '--headless=new', '--disable-gpu', '--no-sandbox',
                           '--virtual-time-budget=180000', '--generate-pdf-document-outline', *args], capture_output=True, text=True)


def main():
    OUT.mkdir(exist_ok=True)
    store = []
    cover_md, body_md, answers = load()
    if answers:
        body_md += '\n\n## เฉลยแบบฝึกหัด\n\n' + '\n\n'.join(f'### {t}\n\n{a}' for t, a in answers) + '\n'
    cover, book_title = cover_html(cover_md, store)
    body, toc, page_css = structure(to_html(body_md, store))
    css = (CSS.replace('__TLWG__', TLWG).replace('__BOOK__', css_str(book_title))
           .replace('__PAGES__', page_css).replace('__W__', str(TEXT_W_MM)))
    htmlp, pdf = OUT / f'{NAME}.html', OUT / f'{NAME}.pdf'

    def write(page_of, cls=''):
        doc = (HEAD % dict(title=html.escape(book_title), css=css, mj=MATHJAX, cls=cls) + cover + toc_html(toc, page_of)
               + (re.sub(r'@@c\d{3}@@', '', body) if page_of else body) + (MEASURE if cls else '') + '</body></html>')
        htmlp.write_text(restore(doc, store), encoding='utf-8')

    write({})                                                           # pass 1: find the pages
    chrome('--no-pdf-header-footer', f'--print-to-pdf={pdf}', htmlp.as_uri())
    text = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], capture_output=True, text=True).stdout
    page_of = {}
    for i, page in enumerate(text.split('\f'), 1):
        for mark in re.findall(r'@@c\d{3}@@', page):
            page_of.setdefault(mark, i)
    missing = [t for mark, t, _ in toc if mark not in page_of]
    write(page_of)                                                      # pass 2: the real thing
    chrome('--no-pdf-header-footer', f'--print-to-pdf={pdf}', htmlp.as_uri())

    write(page_of, cls='check')                                         # screen pass at the text width: MathJax errors, overflow
    dom = chrome('--window-size=1200,900', '--dump-dom', htmlp.as_uri()).stdout
    if '--keep-html' in sys.argv:
        write(page_of)
    else:
        htmlp.unlink()
    info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
    pages = re.search(r'Pages:\s+(\d+)', info).group(1)
    err, n = dom.count('mjx-merror'), dom.count('<mjx-container')
    over = re.search(r'data-overflow-count="(\d+)"', dom)
    print(f'{pdf.relative_to(REPO)}: {pages} pages, {pdf.stat().st_size / 1e6:.1f} MB, chapters {sum(k == "ch" for *_, k in toc)}, '
          f'math {n}, mjx-merror {err}, wider-than-page {over.group(1) if over else "?"}, TOC entries without a page {len(missing)}')
    for m in re.finditer(r'<(mjx-container|table|pre|img|svg)[^>]*data-overflow="1"', dom):
        before = re.sub(r'\s+', ' ', re.sub(r'<svg.*?</svg>|<[^>]+>', ' ', dom[max(0, m.start() - 1500):m.start()], flags=re.S))[-90:]
        print(f'  overflow: <{m.group(1)}> after "…{before.strip()}"')
    bad = thai_in_math(store)
    for b in bad:
        print('  Thai inside math: ' + re.sub(r'\s+', ' ', b)[:100])
    # a tone mark painted on the page before its heading (see the h3 rule in CSS); needs Pillow, skipped without it
    specks = []
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from workbook_audit import dots
        specks = dots(pdf, show=False)
        print(f'  specks at the foot of a page: {len(specks)}' + ''.join(f'  p.{n}' for n, *_ in specks))
    except (ImportError, SystemExit):
        print('  specks at the foot of a page: not checked (Pillow missing)')
    if bad or err or missing or specks or not over or over.group(1) != '0':
        sys.exit(1)


if __name__ == '__main__':
    main()
