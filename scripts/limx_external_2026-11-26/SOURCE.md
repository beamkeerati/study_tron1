# Copies of LimX material outside this repository, fetched 2026-11-26

Kept so that the dated claims in `notes/01-repo-map.md` and `notes/02-sdk-vs-sim.md` can be
re-checked offline. Not our text, not edited. Fetched anonymously on 2026-11-26 from the default
branch of each repository on `raw.githubusercontent.com` and from PyPI; **the commits were not
recorded**, so these are "as of 2026-11-26", not pinned like `scripts/deploy_snapshot_2026-11-26/`.
One exception to "default branch": the 4.0.2 wheel is no longer there. `python3/amd64` of
`limxsdk-lowlevel` held `limxsdk-4.0.2-py3-none-any.whl` until 2026-09 (last seen at commit `81884fc2`,
2026-09-09) and holds 4.1.x since; the three `limxsdk-4.0.2_*` files below come from that history.

| File here | Origin | Used by |
| --- | --- | --- |
| `limxsdk-lowlevel_README.md` | `limxdynamics/limxsdk-lowlevel` README | note 02: "typically at 1 kHz"; the SDK ships as a prebuilt library |
| `tron1-mujoco-sim_README.md` | `limxdynamics/tron1-mujoco-sim` README | note 02: its bundled 3.4.x wheels install on a newer interpreter but fail to import |
| `tron1-rl-deploy-python_README.md` | `limxdynamics/tron1-rl-deploy-python` README | note 02: control loop 500 Hz, policy 50 Hz (also in `params.yaml` of the snapshot) |
| `tron1-robot-description_README.md` | `limxdynamics/tron1-robot-description` README | note 01: `WL_P311D` / `WL_P311E`, wheel-legged quadruped, 16 actuated joints, URDF + SRDF |
| `robot-visualization_README.md` | `limxdynamics/robot-visualization` README | note 01: wheel-legged launch files |
| `limxsdk-4.0.2_METADATA.txt`, `_WHEEL.txt` | `*.dist-info` of the 4.0.2 wheel in `limxsdk-lowlevel` (`python3/amd64`) | note 02: version 4.0.2 (the tag `py3-none-any` is what the mujoco-sim README refers to; note 02 does not quote it) |
| `limxsdk-4.0.2_robot_so_NEEDED.txt` | `readelf -d limxsdk/robot/_robot.so \| grep NEEDED` on that wheel | note 02: needs `libpython3.8.so.1.0` |
| `limxsdk-4.1.2_METADATA.txt`, `_WHEEL.txt` | `*.dist-info` of the 4.1.2 x86_64 wheel on PyPI (`cp38-abi3-manylinux_2_29_x86_64`; PyPI also has an aarch64 wheel) | note 02: `abi3`, `Requires-Python >=3.8,<3.14` |
| `limxsdk-4.1.2_robot_so_NEEDED.txt` | the same `readelf` on the 4.1.2 wheel | note 02: no libpython dependency |

The wheels themselves are not kept (binaries). They were unzipped and inspected with `readelf`
only; nothing from them was executed.
