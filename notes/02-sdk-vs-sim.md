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

There is no `mj_data` on hardware. The motors sit on a CAN/EtherCAT bus behind
LimX's embedded motion controller — closed source, running a real-time OS on a
separate board. The only way in is Ethernet packets in LimX's own wire format.

`limxsdk-lowlevel` is the client library that wraps that wire format into
function calls. It is **not** a controller and **not** a physics engine — it is a
driver, the same role `unitree_sdk2` plays for the Go2.

```
without SDK:  open UDP socket -> pack [header, seq, 12x(q, dq, tau, kp, kd)]
              -> send -> parse the state packet coming back -> repeat at 1 kHz

with SDK:     robot.publishRobotCmd(cmd)
              robot.subscribeRobotState(callback)
```

## The trick: the simulator answers the same protocol

`tron1-mujoco-sim/simulator.py` does this at line 126:

```python
robot = Robot(RobotType.PointFoot, True)   # True = server mode
```

It opens the SDK **as a server** and speaks the robot's protocol. So:

```
policy code ──SDK──► 127.0.0.1     tron1-mujoco-sim   (today)
policy code ──SDK──► 10.192.1.2    real TRON1         (tomorrow)
                                    ^ only the IP changes
```

Both sides of the link use the SDK, even with no hardware in the room.

## Why they designed it this way

Writing `mj_data.ctrl[:] = ...` directly means rewriting the whole I/O layer at
deployment time, and every mismatch — loop rate, latency, joint ordering, units,
whether the PD runs on the motor or in your code — surfaces for the first time on
a robot that can fall over. Putting the simulator on the far side of a socket
exercises the entire deployment path while still in simulation.

## Practical consequence

| Task | SDK needed? |
|---|---|
| Train in Isaac Lab, look at rewards, export ONNX | **No** |
| Run the policy in MuJoCo through the deploy code | **Yes** |
| Run on real hardware | **Yes** |

The SDK wheel (`limxsdk 4.0.2`) is a CPython extension linked against
libpython3.8, so keep it in its own conda env, away from Isaac Lab's 3.10.
