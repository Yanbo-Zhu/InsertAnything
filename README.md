# InsertAnything

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac/sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.2.1-76B900)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.24511-b31b1b.svg)](https://arxiv.org/abs/2609.24511)
[![Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/tOMRmA/InsertAnything-checkpoints)

InsertAnything is an Isaac Lab project for force-conditioned, contact-rich peg-in-hole insertion in simulation and on a real Franka robot. It targets tight-clearance precision insertion, including Tol. IV settings with nominal clearance down to 0.02 mm.

The repository provides:

- Isaac Lab direct-RL environments for five fixed single-hole geometries.
- RL-Games training and evaluation scripts.
- Compact fingertip force feedback, reward shaping, and force logging.
- Stable force-conditioned training with force-signal smoothing and state-independent policy standard deviation.
- Real-robot closed-loop deployment utilities for calibrated Franka experiments.
- Validated single-hole checkpoints hosted on Hugging Face.

The codebase accompanies the paper:

**[InsertAnything: Generalizable Contact-Rich Precision Insertion from Simulation to Reality](https://arxiv.org/abs/2609.24511)**

<p align="center">
  <img src="docs/figures/insertanything_overview.png" alt="InsertAnything overview" width="88%">
</p>

**Framework overview (Fig. 1 in the paper).** Robot setup, standard geometries and clearances, simulation training, and force-conditioned real-world deployment.

<p align="center">
  <img src="docs/figures/insertanything_reward_design.png" alt="Decoupled gated reward design" width="78%">
</p>

**Decoupled gated reward (Fig. 2 in the paper).** Planar error, yaw error, and vertical error are represented separately. As alignment improves, gating increases the reward for approaching the final insertion target.

<p align="center">
  <img src="docs/figures/insertanything_method.png" alt="Stable force-feedback policy learning" width="78%">
</p>

**Stable force-feedback learning (Fig. 3 in the paper).** Exponential-moving-average force smoothing and a state-independent policy standard deviation stabilize force-conditioned policy learning.

## Highlights

- **Simulation-trained, real-robot deployed.** Policies are trained in Isaac Lab and deployed on a real Franka robot through calibrated target poses and deployable robot observations.
- **Force-conditioned insertion.** A compact three-axis force signal supports local contact-aware correction under target-pose bias and tight clearances.
- **Stable contact-rich RL.** Force-signal smoothing and state-independent policy standard deviation improve training stability when force observations change during contact transitions.
- **Single-hole task interface.** Each environment contains one fixed hole and one held peg, with geometry, tolerance, randomization, and observation settings defined explicitly in the task configuration.

## Project status

InsertAnything is organized as an external Isaac Lab extension. The repository includes the insertion assets, task configurations, RL-Games configurations, and deployment utilities. The Franka robot and workbench models are loaded from the standard Isaac Lab and Isaac Sim asset collections.

## Installation

Install NVIDIA Isaac Sim and Isaac Lab first. Follow the official Isaac Lab repository and installation guide:

- [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)

The release was validated with Isaac Sim 5.0, Isaac Lab 2.2.1, and Python 3.11.

Then install InsertAnything:

```bash
python -m pip install -e source/InsertAnything
```

All commands in this repository use Bash syntax and are intended to be run from the repository root on Linux.

List the registered InsertAnything environments:

```bash
python scripts/list_envs.py
```

## Simulation experiments

Simulation tasks, reward settings, observation schemas, contact-force options, training commands, and evaluation commands are documented in:

- [Single-hole task README](source/InsertAnything/InsertAnything/tasks/direct/insertanything/README.md)

Example training command:

```bash
python scripts/rl_games/train.py --task InsertAnything-CircleHole-I-Direct-v0 --num_envs 128 --headless
```

Example evaluation command:

```bash
python scripts/rl_games/play.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 128 \
  --headless \
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth
```

Evaluation writes TensorBoard events automatically. Each run has a unique directory under
`logs/rl_games/InsertAnything/play/<name>-<timestamp>/tensorboard/`. Override the root with `--log_dir`.
The absolute run directory and TensorBoard command are printed at startup.

```bash
python scripts/rl_games/play.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 8 --max_episodes 50 --seed 0 --headless \
  --log_dir logs/rl_games/InsertAnything/play \
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth

tensorboard --logdir logs/rl_games/InsertAnything/play --port 6006
```

Open `http://localhost:6006`. Metrics use the completed parallel evaluation round as their horizontal axis.
They include per-round and cumulative success rates (percent), success counts, mean return, and mean first-success
time (seconds). Cumulative success time is weighted by successful episodes. A zero-success round omits its mean
success time. `reward_terms_raw/*` contains time averages of the existing unscaled reward terms, not their weighted
contributions to return. `progress/*` records rounds, control steps and cumulative individual environment episodes.
For synchronized tasks, 8 environments and 50 rounds correspond to 400 individual environment episodes.

To also record the same metrics in an online W&B run:

```bash
python scripts/rl_games/play.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 8 --max_episodes 50 --seed 0 --headless \
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth \
  --track --wandb-entity dylan1000-bht \
  --wandb-project-name Isaac-InsertAnything \
  --wandb-name play-LHole-III --wandb-mode online
```

To record W&B data offline without uploading:

```bash
python scripts/rl_games/play.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 8 --max_episodes 50 --seed 0 --headless \
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth \
  --track --wandb-project-name Isaac-InsertAnything \
  --wandb-name play-LHole-III-offline --wandb-mode offline
```

W&B is optional and is not initialized without `--track`. The project defaults to `Isaac-InsertAnything`; entity
uses the supplied argument or the SDK's account configuration. Run names use `--wandb-name`, then the agent's
`full_experiment_name`, then the task name. `--wandb-mode offline` saves W&B data locally without uploading;
if the mode is omitted, the SDK follows `WANDB_MODE` and its own defaults. W&B local files share the evaluation
run directory. Online runs print their URL; offline runs explicitly print the local path. Requested W&B logging
failures stop evaluation rather than silently falling back. TensorBoard is written directly and is not automatically
synced to W&B a second time. Completed rounds are flushed immediately; normal exit and Ctrl+C close the recorders
before shutting down the simulator.

Video behavior is unchanged: `--video` still ends playback at `--video_length` (default 200 steps).
If playback ends before the first completed evaluation round, the event file contains configuration but no round
curves. Use evaluation without video when collecting full multi-round statistics. Terminal output saved by `tee`
is a separate text log, not a TensorBoard event file. No training losses are produced during evaluation.

Registered simulation tasks:

| Gym ID | Description |
| --- | --- |
| `InsertAnything-CircleHole-I-Direct-v0` | Circle hole, Tol. I |
| `InsertAnything-SquareHole-II-Direct-v0` | Square hole, Tol. II |
| `InsertAnything-LHole-III-Direct-v0` | L-shape hole, Tol. III |
| `InsertAnything-TriangleHole-IV-Direct-v0` | Triangle hole, Tol. IV |
| `InsertAnything-HexagonHole-III-Direct-v0` | Hexagon hole, Tol. III with fingertip force feedback |
| `InsertAnything-HexagonHole-IV-Direct-v0` | Hexagon hole, Tol. IV |

## Pretrained checkpoints

Optional checkpoints are hosted on Hugging Face:

- [InsertAnything checkpoints](https://huggingface.co/tOMRmA/InsertAnything-checkpoints)

Download the released checkpoints:

```bash
hf download tOMRmA/InsertAnything-checkpoints InsertAnything-LHole-III-Direct-v0.pth \
  --repo-type model \
  --local-dir source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints

hf download tOMRmA/InsertAnything-checkpoints InsertAnything-HexagonHole-III-Direct-v0.pth \
  --repo-type model \
  --local-dir source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints
```

Checkpoint filenames, hashes, observation dimensions, and evaluation commands are documented in the [checkpoint README](source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/README.md).

Large checkpoint binaries are intentionally ignored by git.

## Real-robot deployment

The real-robot deployment code lives in:

```text
source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real/
```

The main multi-episode closed-loop runner is:

```text
source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real/run_multi_episode_closed_loop.py
```

It requires a calibrated target pose, a compatible Franka control server, and the sensor/control interfaces described in [the sim-to-real README](source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real/README.md).

## Citation

If you find InsertAnything useful in your research, please cite:

```bibtex
@misc{ma2026insertanythinggeneralizablecontactrichprecision,
  title         = {InsertAnything: Generalizable Contact-Rich Precision Insertion from Simulation to Reality},
  author        = {Zhenghua Ma and Xinpan Meng and Zeyu Liu and Muyuan Ma and Hengdi Zhang and Houcheng Li and Long Cheng},
  year          = {2026},
  eprint        = {2609.24511},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.24511}
}
```

## License

InsertAnything is released under the BSD 3-Clause license. See [LICENSE](LICENSE).
