# notes/

Understanding, not instructions. Setup lives in the top-level `README.md`.

| File | Contents |
|---|---|
| [01-repo-map.md](01-repo-map.md) | The 12 TRON1 repos at `github.com/limxdynamics` and how they stack |
| [02-sdk-vs-sim.md](02-sdk-vs-sim.md) | What `limxsdk-lowlevel` is, and why the simulator impersonates the robot over a socket |
| [03-sf-task-anatomy.md](03-sf-task-anatomy.md) | `Isaac-Limx-SF-Blind-Flat-v0` pulled apart: obs/action/reward/randomisation/PPO |
| [04-file-structure-vs-ros2.md](04-file-structure-vs-ros2.md) | Isaac Sim vs Isaac Lab vs project, mapped onto ROS 2 habits, and where to change what |
| [05-rl-fundamentals-ground-up.md](05-rl-fundamentals-ground-up.md) | RL from zero, in Thai: MDP as a 5-slot form, Markov property and why violating it breaks everything, what the policy network physically is (nodes, layers, why no output activation), the full action pipeline μ → sample → ×0.25 → PD → torque, `act` vs `act_inference`, step/episode/iteration/batch, why 24 steps per iteration, then Bellman → MC/TD → actor-critic → GAE → PPO — every term tied back to real config values and line numbers |
