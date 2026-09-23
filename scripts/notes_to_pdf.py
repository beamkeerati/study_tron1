#!/usr/bin/env python3
"""Render notes/*.md to notes/pdf/*.pdf.

Pipeline: python-markdown (tables, fenced_code) -> HTML in Noto Sans Thai /
Noto Sans Mono -> every $...$ / $$...$$ left as LaTeX and typeset by the
MathJax vendored at notes/diagrams/mathjax/tex-svg.js (offline) -> headless
Chrome --print-to-pdf (A4, no header/footer). Each build also dumps the DOM
and counts mjx-merror; anything other than 0 means a formula did not parse.

Usage (from anywhere):
    python3 scripts/notes_to_pdf.py                 # 00, 05, 06, 07
    python3 scripts/notes_to_pdf.py 05-rl-fundamentals-ground-up 00-glossary

Needs: pip install --user markdown ; google-chrome ; fonts-noto (Thai, Mono).
"""
import re, sys, html, subprocess, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
NOTES = REPO / 'notes'; OUT = NOTES / 'pdf'; OUT.mkdir(exist_ok=True)
MATHJAX = (NOTES / 'diagrams/mathjax/tex-svg.js').as_uri()
import markdown

def protect(src):
    store = []
    def keep(m):
        store.append(m.group(0)); return f'MATHPH{len(store)-1:05d}ZZ'
    src = re.sub(r'```.*?```', keep, src, flags=re.S)          # fenced code
    src = re.sub(r'\$\$.*?\$\$', keep, src, flags=re.S)        # display math
    src = re.sub(r'(?<!\\)\$(?!\s)(?:\\.|[^$\\])+?\$', keep, src)  # inline math
    return src, store

def restore(h, store):
    def back(m):
        s = store[int(m.group(1))]
        if s.startswith('```'):
            body = s.split('\n', 1)[1].rsplit('```', 1)[0]
            return f'<pre><code>{html.escape(body)}</code></pre>'
        if s.startswith('$$'):
            return r'\[' + html.escape(s[2:-2]) + r'\]'
        return r'\(' + html.escape(s[1:-1]) + r'\)'
    return re.sub(r'MATHPH(\d{5})ZZ', back, h)

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Noto Sans Thai', 'Noto Sans', sans-serif; font-size: 10.5pt; line-height: 1.55; color: #111; }
h1 { font-size: 18pt; margin-top: 1.2em; } h1:not(.top) { page-break-before: always; }
h2 { font-size: 14pt; margin-top: 1.4em; border-bottom: 1px solid #bbb; padding-bottom: 2px; }
h3 { font-size: 12pt; margin-top: 1.2em; } h4 { font-size: 11pt; }
code, pre { font-family: 'Noto Sans Mono', monospace; font-size: 8.8pt; }
code { background: #f2f2f2; padding: 0 3px; border-radius: 3px; }
pre { background: #f5f5f5; padding: 8px; white-space: pre-wrap; word-break: break-word; border: 1px solid #ddd; }
table { border-collapse: collapse; font-size: 8.8pt; margin: 8px 0; width: 100%; }
th, td { border: 1px solid #bbb; padding: 3px 5px; vertical-align: top; }
th { background: #eee; }
blockquote { border-left: 3px solid #999; margin: 8px 0; padding: 2px 10px; color: #333; background: #fafafa; }
mjx-container[display="true"] { margin: 6px 0 !important; overflow-x: auto; }
tr, pre, blockquote { page-break-inside: avoid; }
a { color: #1a4d8f; text-decoration: none; }
"""
HEAD = """<!doctype html><html lang="th"><head><meta charset="utf-8"><title>%s</title><style>%s</style>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']],packages:{'[+]':['ams','boldsymbol']}},svg:{fontCache:'global'}};</script>
<script src="%s"></script></head><body>"""

def build(name):
    src = (NOTES / f'{name}.md').read_text(encoding='utf-8')
    p, store = protect(src)
    h = markdown.markdown(p, extensions=['tables', 'fenced_code', 'sane_lists'])
    h = restore(h, store)
    h = h.replace('<h1>', '<h1 class="top">', 1)
    htmlp = OUT / f'{name}.html'
    htmlp.write_text(HEAD % (name, CSS, MATHJAX) + h + '</body></html>', encoding='utf-8')
    pdf = OUT / f'{name}.pdf'
    subprocess.run(['google-chrome', '--headless=new', '--disable-gpu', '--no-sandbox',
                    '--virtual-time-budget=120000', '--no-pdf-header-footer',
                    f'--print-to-pdf={pdf}', htmlp.as_uri()], check=True, capture_output=True)
    dom = subprocess.run(['google-chrome', '--headless=new', '--disable-gpu', '--no-sandbox',
                          '--virtual-time-budget=120000', '--dump-dom', htmlp.as_uri()],
                         capture_output=True, text=True).stdout
    err = dom.count('mjx-merror'); n = dom.count('<mjx-container')
    htmlp.unlink()
    print(f'{pdf.relative_to(REPO)}: {pdf.stat().st_size/1e6:.1f} MB, mathjax containers {n}, mjx-merror {err}')
    if err: sys.exit(f'{name}: {err} MathJax errors')

for n in sys.argv[1:] or ['00-glossary', '05-rl-fundamentals-ground-up', '06-training-walkthrough-bridge', '07-compute-at-every-level']:
    build(n)
