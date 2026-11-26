# 02 — What the SDK is, and why the simulator pretends to be a robot

## The confusion this note resolves

In MuJoCo/mjlab you write control directly:

```python
obs    = get_obs(mj_data)
action = policy(obs)
mj_data.ctrl[:] = kp * (action - mj_data.qpos[7:]) - kd * mj_data.qvel[6:]
mujoco.mj_step(mj_model, mj_data)
```

Policy and physics share one process, so there is nothing to "connect" to.

## Why the real robot cannot work that way

There is no `mj_data` on hardware. What sits between the network port and the
motors — the bus, the embedded controller, its operating system — is not documented
in any repo read for these notes. The only way in is the network, through LimX's
SDK, which ships as a prebuilt library without source.

`limxsdk-lowlevel` is the library that wraps that link into function calls. It is
**not** a controller and **not** a physics engine — it is a driver, the same role
`unitree_sdk2` plays for the Go2. The comparison stops at one point: this SDK is
also what the simulator links against to play the robot's side (next section).

```
without SDK:  connect -> serialise 8x(mode, q, dq, tau, Kp, Kd)   # SF: 8 joints
              -> send -> parse the state coming back -> repeat every loop tick

with SDK:     robot.publishRobotCmd(cmd)
              robot.subscribeRobotState(callback)
```

The "without SDK" lines illustrate the work the SDK hides; they are not a
description of the wire format, which is not published. The joint count and the
fields are those of the SF deploy program (`params.yaml:3-11`,
`SolefootController.py:36-41` in `scripts/deploy_snapshot_2026-11-26/`). That
program runs its loop at 500 Hz with the policy at 50 Hz (`params.yaml:2`, `:33`);
the SDK's own README says commands are "typically" sent at 1 kHz (read 2026-11-26; the READMEs and wheel
metadata this note quotes are copied in `scripts/limx_external_2026-11-26/`).

## The trick: the simulator answers the same protocol

`tron1-mujoco-sim/simulator.py` does this at line 126:

```python
robot = Robot(RobotType.PointFoot, True)   # True = this process plays the robot's side
```

With that flag the simulator takes the robot's side of the link: it receives
commands and publishes state (`subscribeRobotCmdForSim`, `publishRobotStateForSim`
— `simulator.py:55`, `:87`). These notes call this *server mode*. So:

```
policy code ──SDK──► 127.0.0.1     tron1-mujoco-sim   (today)
policy code ──SDK──► 10.192.1.2    real TRON1         (tomorrow)
                                    ^ only the IP changes
```

Both sides of the link use the SDK, even with no hardware in the room.

The IP is the only thing that changes in the code you write, not the only thing
that changes at run time: with any address other than `127.0.0.1` the controller
does not start by itself (`main.py:39`). It waits for the robot's calibration state
and for L1 + Y on the joystick (`SolefootController.py:479-481`).

## Why they designed it this way

Writing `mj_data.ctrl[:] = ...` directly means rewriting the whole I/O layer at
deployment time, and every mismatch — loop rate, latency, joint ordering, units,
where the PD is evaluated — surfaces for the first time on
a robot that can fall over. Putting the simulator on the far side of a socket
exercises the entire deployment path while still in simulation.

## Practical consequence

| Task | SDK needed? |
|---|---|
| Train in Isaac Lab, look at rewards, export ONNX | **No** |
| Run the policy in MuJoCo through the deploy code | **Yes** |
| Run on real hardware | **Yes** |

The 4.0.2 SDK wheel (the one in `limxsdk-lowlevel` when this note was written) is a
CPython extension that needs `libpython3.8.so.1.0` — checked with `readelf -d` on its
`_robot.so`, 2026-11-26 — so it is tied to Python 3.8 (the README of `tron1-mujoco-sim` says of its own bundled 3.4.x
wheels that they install on a newer interpreter but fail to import) and has to live in its own
conda env, away from Isaac Lab's 3.10. PyPI has since published 4.1.2, an `abi3` wheel
(`Requires-Python >=3.8,<3.14`) with no libpython dependency (checked 2026-11-26): check which one you have
before setting up a second env.
