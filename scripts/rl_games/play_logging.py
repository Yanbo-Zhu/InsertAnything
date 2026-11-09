"""Evaluation metrics and logging without simulator dependencies."""

import importlib
import json
import re
import shlex
from datetime import datetime
from pathlib import Path

import torch


class EvaluationMetrics:
    """Aggregate synchronous InsertAnything evaluation rounds, including terminal rewards."""

    def __init__(self):
        self.round = 0
        self.control_steps = 0
        self.successes = 0
        self.episodes = 0
        self.success_time_sum_s = 0.0
        self.returns = None
        self.term_sums = {}
        self.term_counts = {}

    def step(self, rewards, infos):
        self.control_steps += 1
        rewards = rewards.detach().reshape(-1)
        if self.returns is None:
            self.returns = torch.zeros_like(rewards)
        self.returns += rewards
        for key, value in infos.items():
            if key.startswith("logs_rew_"):
                value = value.detach() if isinstance(value, torch.Tensor) else float(value)
                self.term_sums[key] = self.term_sums.get(key, 0.0) + value
                self.term_counts[key] = self.term_counts.get(key, 0) + 1

        result = infos.get("eval_metrics")
        if result is None:
            return None
        self.round += 1
        successes = int(result["success_count"])
        episodes = int(result["episode_count"])
        success_time_sum_s = float(result["success_time_sum_s"])
        self.successes += successes
        self.episodes += episodes
        self.success_time_sum_s += success_time_sum_s
        metrics = {
            "progress/round": self.round,
            "progress/control_steps": self.control_steps,
            "progress/environment_episodes": self.episodes,
            "eval/success_count": successes,
            "eval/episode_count": episodes,
            "eval/success_rate_percent": 100.0 * successes / episodes,
            "eval/cumulative_success_count": self.successes,
            "eval/cumulative_success_rate_percent": 100.0 * self.successes / self.episodes,
            "eval/mean_return": self.returns.mean().item(),
        }
        if successes:
            metrics["eval/mean_success_time_s"] = success_time_sum_s / successes
        if self.successes:
            metrics["eval/cumulative_mean_success_time_s"] = self.success_time_sum_s / self.successes
        for key, total in self.term_sums.items():
            mean = total / self.term_counts[key]
            metrics[f"reward_terms_raw/{key.removeprefix('logs_rew_')}"] = float(mean)
        self.returns.zero_()
        self.term_sums.clear()
        self.term_counts.clear()
        return metrics


class PlayLogger:
    """Write identical round metrics to TensorBoard and, optionally, W&B."""

    def __init__(self, root, name, config, *, track=False, entity=None, project="Isaac-InsertAnything", mode=None):
        from torch.utils.tensorboard import SummaryWriter

        root = Path(root).expanduser().resolve()
        safe_name = re.sub(r"[\\/]", "_", name)
        self.directory = root / f"{safe_name}-{datetime.now():%Y%m%d-%H%M%S-%f}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.writer = SummaryWriter(str(self.directory / "tensorboard"))
        self.run = None
        print(f"[INFO] Evaluation logs: {self.directory}")
        print(f"[INFO] TensorBoard: tensorboard --logdir {shlex.quote(str(root))}")
        try:
            config = json.loads(json.dumps(config, default=str))
            self.writer.add_text("run/config", json.dumps(config, indent=2), 0)
            self.writer.flush()
            if track:
                wandb = importlib.import_module("wandb")
                self.run = wandb.init(
                    entity=entity,
                    project=project,
                    name=name,
                    dir=str(self.directory),
                    config=config,
                    job_type="play",
                    mode=mode,
                    sync_tensorboard=False,
                    settings=wandb.Settings(init_timeout=60),
                )
                if self.run is None or self.run.settings.mode == "disabled":
                    raise RuntimeError("W&B tracking was requested but is disabled.")
                self.run.define_metric("progress/round")
                self.run.define_metric("*", step_metric="progress/round")
                if self.run.offline:
                    print(f"[INFO] W&B offline data (not uploaded): {self.run.dir}")
                else:
                    print(f"[INFO] W&B run: {self.run.url}")
        except BaseException:
            self.close(exit_code=1)
            raise

    def log(self, metrics):
        step = int(metrics["progress/round"])
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
        self.writer.flush()
        if self.run is not None:
            try:
                self.run.log(metrics, step=step)
            except Exception as exc:
                raise RuntimeError("W&B metric logging failed; stopping evaluation.") from exc

    def close(self, exit_code=0):
        writer, self.writer = self.writer, None
        run, self.run = self.run, None
        try:
            if writer is not None:
                try:
                    writer.flush()
                finally:
                    writer.close()
        finally:
            if run is not None:
                run.finish(exit_code=exit_code)
