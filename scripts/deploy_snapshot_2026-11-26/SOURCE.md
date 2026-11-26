# Snapshot of LimX deploy-side files, fetched 2026-11-26

Read-only copies, kept so that chapter 30 of the workbook can cite line numbers that stay
checkable. Not our code, not edited, never executed here — `scripts/workbook_answer_key.py 30`
only reads them as text.

| File here | Origin | Commit (committer date) |
| --- | --- | --- |
| `params.yaml` | `limxdynamics/tron1-rl-deploy-python` · `controllers/model/SF_TRON1A/params.yaml` | `f15f783` (2026-11-26) |
| `SolefootController.py` | same repo · `controllers/SolefootController.py` | `f15f783` |
| `main.py` | same repo · `main.py` | `f15f783` |
| `simulator.py` | `limxdynamics/tron1-mujoco-sim` · `simulator.py` | `988123a` (2026-11-05) |
| `onnx_shapes.json` | shapes read with `onnx.load` from `controllers/model/SF_TRON1A/policy/{isaaclab,isaacgym}/{policy,encoder}.onnx` at `f15f783`; the `.onnx` files themselves are not kept | `f15f783` |

Both repos are Apache-2.0 (their `LICENSE` files). Fetched anonymously from
`raw.githubusercontent.com` at the pinned commits.
