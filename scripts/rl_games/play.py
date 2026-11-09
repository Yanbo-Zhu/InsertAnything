# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RL-Games checkpoint in an InsertAnything environment."""

import argparse
import signal
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during evaluation.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rl_games_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--log_dir", type=str, default=None, help="Root directory for evaluation logs.")
parser.add_argument(
    "--track",
    type=lambda value: bool(strtobool(value)),
    nargs="?",
    const=True,
    default=False,
    help="Record evaluation metrics with Weights & Biases as well as TensorBoard.",
)
parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity or team.")
parser.add_argument("--wandb-project-name", type=str, default="Isaac-InsertAnything", help="Weights & Biases project.")
parser.add_argument("--wandb-name", type=str, default=None, help="Evaluation run name.")
parser.add_argument("--wandb-mode", choices=("online", "offline"), default=None, help="Override W&B environment mode.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=None,
    help="Stop after this many complete evaluation episodes. Runs indefinitely when omitted.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import InsertAnything.tasks as insertanything_tasks
import isaaclab_tasks
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from play_logging import EvaluationMetrics, PlayLogger

_REGISTERED_TASK_MODULES = (isaaclab_tasks, insertanything_tasks)
_stop_requested = False


def _request_stop(_signum, _frame):
    # A KeyboardInterrupt raised inside a native physics callback can be swallowed by Kit.
    global _stop_requested
    _stop_requested = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    if hasattr(env_cfg, "evaluation_mode"):
        env_cfg.evaluation_mode = True

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env_cfg.scene.num_envs
    run_name = args_cli.wandb_name or agent_cfg["params"]["config"].get("full_experiment_name") or args_cli.task
    evaluation_logger = PlayLogger(
        args_cli.log_dir or os.path.join(log_root_path, "play"),
        run_name,
        {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "seed": env_cfg.seed,
            "num_envs": env_cfg.scene.num_envs,
            "max_episodes": args_cli.max_episodes,
            "video": args_cli.video,
            "video_length": args_cli.video_length,
            "env_cfg": env_cfg.to_dict(),
            "agent_cfg": agent_cfg,
        },
        track=args_cli.track,
        entity=args_cli.wandb_entity,
        project=args_cli.wandb_project_name,
        mode=args_cli.wandb_mode,
    )
    env = None
    exit_code = 0
    try:
        if _stop_requested:
            print("[INFO] Evaluation interrupted by user.")
            return
        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_run_dir = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            video_folder = os.path.join(log_dir, "videos", "play", video_run_dir)
            video_kwargs = {
                "video_folder": video_folder,
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print(f"[INFO] Recording video to: {video_folder}")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rl-games
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

        # register the environment to rl-games registry
        # note: in agents configuration: environment name must be "rlgpu"
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        # load previously trained model
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

        # set number of actors into agent config
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        # create runner from rl-games
        runner = Runner()
        runner.load(agent_cfg)
        # obtain the agent from the runner
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

        dt = env.unwrapped.step_dt

        # reset environment. Discard the first reset to avoid transient simulator/cache state
        # from freshly spawned assets or sensors affecting the first evaluation episode.
        _ = env.reset()
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        timestep = 0
        completed_episodes = 0
        evaluation_metrics = EvaluationMetrics()
        # required: enables the flag for batched observations
        _ = agent.get_batch_size(obs, 1)
        # initialize RNN states if used
        if agent.is_rnn:
            agent.init_rnn()
        # simulate environment
        # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
        #   attempt to have complete control over environment stepping. However, this removes other
        #   operations such as masking that is used for multi-agent learning by RL-Games.
        while simulation_app.is_running() and not _stop_requested:
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # convert obs to agent format
                obs = agent.obs_to_torch(obs)
                # agent stepping
                actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
                # env stepping
                obs, rewards, dones, infos = env.step(actions)

                metrics = evaluation_metrics.step(rewards, infos)
                if metrics is not None:
                    evaluation_logger.log(metrics)

                if isinstance(infos, dict) and "eval_printout" in infos:
                    print(infos["eval_printout"], end="")
                    completed_episodes += 1

                # perform operations for terminated episodes
                if len(dones) > 0:
                    # reset rnn state for terminated episodes
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0
            if args_cli.video:
                timestep += 1
                # exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            if args_cli.max_episodes is not None and completed_episodes >= args_cli.max_episodes:
                break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
        if _stop_requested:
            print("[INFO] Evaluation interrupted by user.")
    except Exception:
        exit_code = 1
        raise
    finally:
        try:
            evaluation_logger.close(exit_code=exit_code)
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    # Flush logs before Kit's default Ctrl+C handler unloads simulator plugins.
    signal.signal(signal.SIGINT, _request_stop)
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] Evaluation interrupted by user.")
    finally:
        simulation_app.close()
