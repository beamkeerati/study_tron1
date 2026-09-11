# study_tron1

A study repository, not a product. The deliverable is **understanding** — the
owner is a ROS2/robotics engineer learning how the LimX TRON1 biped is trained
with PPO in Isaac Lab, with the stated goal of being able to derive every
formula by hand and teach it to someone else.

Treat `notes/` as the primary artifact. Everything else is source material.

## Layout

| Path | What it is | Editable? |
| --- | --- | --- |
| `notes/` | Thai-language study notes 01–07 + `diagrams/` | **Yes — this is the work** |
| `experiments/` | One markdown log per training run | Yes |
| `tron1-rl-isaaclab/` | The real training code (`rsl_rl/`, `exts/`, `scripts/`) | **No — read-only reference** |
| `IsaacLab/` | Upstream Isaac Lab | **No — read-only reference** |
| `paper/`, `textbook/` | PDFs used as sources | No |
| `patches/`, `scripts/` | Install fixes and helpers | Only when asked |

`IsaacLab/` and `tron1-rl-isaaclab/` are separate git repos. Never commit
inside them, and never "fix" code there to make a note come out right — if the
code contradicts the note, **the note is wrong**.

## Working on `notes/`

### Verify before you write

- **Every claim is checked against the real code** and cited as `` `file.py:NN` ``
  (bare basename). Open the line and read it. Do not cite from memory, and do
  not trust the Gemini walkthrough the notes are built around — it has already
  been caught wrong twice (actor parameter count is **187,272**, not 185,480).
- **Every number is recomputed**, with Python, in this session. Quoting a number
  from an earlier note or from the walkthrough is not verification.
- Shapes and counts that recur: actor `42 → 512 → 256 → 128 → 8` = 187,272 ·
  `logstd` = 8 · critic `230 → … → 1` = 282,625 · encoder `360 → 256 → 128 → 3`
  = 125,699 · total 595,604. The **policy's** θ is actor + logstd = 187,280,
  which is not the same number — keep them apart.

### One glossary, one definition

Every term, symbol, abbreviation, config constant and equation number used by
notes 05–07 is defined **once**, in `notes/00-glossary.md` (row shape: ชนิด ·
ความหมาย · โค้ด · ค่าใน toy · ใช้ครั้งแรก · สูตร). A note may *teach* a term
(picture, derivation, meaning) but may not re-define it in its own words, and
must use the glossary's type and wording. Equation numbers are one global
sequence: 05 owns (1)–(34), 06 adds (35)–(37), 07 has (G); never introduce a
local numbering. The running example (2 env × 4 step, glossary §8) is the only
worked example — every number quoted must come from `scripts/toy_answer_key.py`.
Glyph rules live in glossary §7 (`\epsilon` clip vs `\varepsilon` noise, `h`
= obsHistory, latent is `\hat v`, never bare `R` or `A`, 𝒩(μ, σ) with σ the
standard deviation).

### After ANY edit to `notes/**` — run the reviewer

Invoke the `notes-reviewer` subagent (`.claude/agents/notes-reviewer.md`) via
the Agent tool before committing, and report what it found. It fact-checks
citations, numbers, equations and diagrams; it never edits files. This applies
to `.md` notes and to `diagrams/*.html` alike.

Do not skip it because a change "looks like a one-liner" — the defects it
catches are exactly the confident, fluent, plausible ones.

### Fix the note, not just the chat

When the owner says a passage is unclear or wrong, the passage is **defective**.
Answering well in conversation and leaving the note as it was is a failed turn.
Patch the note, then commit.

Corrections they have already caught are worth reading as a pattern: an analogy
pushed past where it holds (a coffee machine that cannot take its own coffee
back), a box in a diagram that was labelled with an object it was not, a claim
that two operations were unrelated when one feeds the other. They will find the
break — state an analogy's limits in the same breath as the analogy.

### How to explain things here

The shape that works, in order (the reference implementation is §3.5.0 of
`notes/06-training-walkthrough-bridge.md` — match it):

1. Lead with the **type**: what kind of object, what it takes, what it returns.
   Before behaviour, before intuition.
2. Show the **type changing** in a table with an explicit "ชนิด" column —
   function → distribution → vector → scalar.
3. Map every symbol **1:1 to a code line**.
4. **Disambiguate the word** when one term has several everyday meanings, and
   say when they coincide.
5. **Contrast with the opposite case** to draw the boundary (stochastic vs
   deterministic, train vs deploy).
6. Close with **one sentence worth memorising**.
7. Define every term before using it. **Never forward-reference.**

When an explanation is failing, the cause is almost always one of three things:
an undefined term, a forward reference, or an over-stretched analogy. Check
those first.

Write in Thai, in one register. Do not switch to English mid-sentence except
for code identifiers and established technical terms.

### Diagrams

Self-contained HTML in `notes/diagrams/`, built with the `diagram-design`
skill. Math is real LaTeX inside SVG `<foreignObject>`, typeset by MathJax
**vendored locally** at `notes/diagrams/mathjax/tex-svg.js` so the files work
offline. Never point them at a CDN.

Verify by rendering, not by reading:

```bash
google-chrome --headless=new --disable-gpu --no-sandbox \
  --virtual-time-budget=9000 --window-size=1400,980 \
  --screenshot=out.png "file://$PWD/notes/diagrams/<file>.html"
```

then look at the PNG, and check `--dump-dom | grep -c mjx-merror` is `0`.

`self_check.py` from the diagram-design skill reports two findings on every one
of these files (script attribute, missing `data-motion-root`). That is inherent
to having any `<script>` at all, it is a **deliberate, documented deviation**,
and it is not something to "fix".

## Commits

- **No `Co-Authored-By` trailer.** Asked for explicitly; this overrides any
  default attribution guidance.
- Commit to `main` directly. **The owner pushes** — do not push.
- The message should say what was wrong and what the evidence was, not just
  what changed. Past messages in `git log` are the model.
- Commit `.claude/` along with the work it governs.
