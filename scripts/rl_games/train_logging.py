"""Training metrics, W&B axes and cleanup helpers, independent of the simulator."""

import sys
from contextlib import contextmanager


class TrainingMetrics:
    """Summarize completed training episodes and raw reward terms per epoch."""

    def __init__(self):
        self.control_steps = 0
        self.episodes = 0
        self.successes = 0
        self.success_time_sum_s = 0.0
        self._clear_epoch()

    def _clear_epoch(self):
        self.epoch_episodes = 0
        self.epoch_successes = 0
        self.epoch_success_time_sum_s = 0.0
        self.return_sum = 0.0
        self.term_sums = {}
        self.term_counts = {}

    def step(self, infos, completed_returns=None):
        self.control_steps += 1
        for name, value in infos.items():
            if name.startswith("logs_rew_"):
                value = value.detach() if hasattr(value, "detach") else float(value)
                # Out-of-place addition leaves the environment's original extras untouched.
                self.term_sums[name] = self.term_sums.get(name, 0.0) + value
                self.term_counts[name] = self.term_counts.get(name, 0) + 1

        result = infos.get("train_episode_metrics")
        if result is None:
            return
        episodes = int(result["episode_count"])
        successes = int(result["success_count"])
        success_time_sum_s = float(result["success_time_sum_s"])
        self.epoch_episodes += episodes
        self.epoch_successes += successes
        self.epoch_success_time_sum_s += success_time_sum_s
        self.episodes += episodes
        self.successes += successes
        self.success_time_sum_s += success_time_sum_s
        # RL-Games maintains these raw returns across rollouts and clears them after process_infos().
        self.return_sum = self.return_sum + completed_returns.detach().sum()

    def end_epoch(self, epoch):
        metrics = {
            "train/epoch": epoch,
            "progress/control_steps": self.control_steps,
            "progress/environment_episodes": self.episodes,
            "train/success_count": self.epoch_successes,
            "train/episode_count": self.epoch_episodes,
            "train/cumulative_success_count": self.successes,
        }
        if self.epoch_episodes:
            metrics["train/success_rate_percent"] = 100.0 * self.epoch_successes / self.epoch_episodes
            metrics["train/mean_return"] = float(self.return_sum) / self.epoch_episodes
        if self.episodes:
            metrics["train/cumulative_success_rate_percent"] = 100.0 * self.successes / self.episodes
        if self.epoch_successes:
            metrics["train/mean_success_time_s"] = self.epoch_success_time_sum_s / self.epoch_successes
        if self.successes:
            metrics["train/cumulative_mean_success_time_s"] = self.success_time_sum_s / self.successes
        for name, total in self.term_sums.items():
            metrics[f"reward_terms_raw/{name.removeprefix('logs_rew_')}"] = float(total / self.term_counts[name])
        self._clear_epoch()
        return metrics


class TrainingMetricsObserver:
    """Add metrics to RL-Games' existing writer while retaining its original observer."""

    def __init__(self, observer):
        self.observer = observer
        self.metrics = TrainingMetrics()
        self.last_epoch = None
        self.writer = None

    def __getattr__(self, name):
        # Includes before_init, after_steps, after_clear_stats and optional PBT hooks.
        return getattr(self.observer, name)

    def after_init(self, algo):
        self.observer.after_init(algo)
        self.algo = algo
        self.writer = algo.writer
        self.metrics = TrainingMetrics()
        self.last_epoch = None

    def process_infos(self, infos, done_indices):
        self.observer.process_infos(infos, done_indices)
        if self.writer is None or not isinstance(infos, dict):
            return
        completed_returns = None
        if "train_episode_metrics" in infos:
            completed_returns = self.algo.current_rewards[done_indices.reshape(-1)]
        self.metrics.step(infos, completed_returns)

    def after_print_stats(self, frame, epoch_num, total_time):
        self.observer.after_print_stats(frame, epoch_num, total_time)
        # Some RL-Games training paths invoke this hook twice for the same epoch.
        if self.writer is None or epoch_num == self.last_epoch:
            return
        self.last_epoch = epoch_num
        for name, value in self.metrics.end_epoch(epoch_num).items():
            self.writer.add_scalar(name, value, frame)
        self.writer.flush()


def close_preserving_error(callback, resource):
    """Run cleanup without replacing an exception already being propagated."""
    original_error = sys.exception()
    try:
        callback()
    except BaseException as cleanup_error:
        if original_error is None:
            raise
        note = f"Cleanup failed for {resource}: {type(cleanup_error).__name__}: {cleanup_error}"
        original_error.add_note(note)
        print(f"[ERROR] {note}", file=sys.stderr)


@contextmanager
def cleanup_on_exit(callback, resource):
    """Always attempt cleanup, including during initialization failures."""
    try:
        yield
    finally:
        close_preserving_error(callback, resource)


def close_training_writer(runner):
    """Flush RL-Games' asynchronous TensorBoard writer before finishing W&B."""
    writer = getattr(getattr(runner, "algo_observer", None), "writer", None)
    if writer is not None:
        writer.close()


def configure_wandb_axes(run):
    """Set training chart axes without replacing W&B's internal history counter."""
    for pattern in ("train/*", "reward_terms_raw/*", "performance/*"):
        run.define_metric(pattern, step_metric="info/epochs", step_sync=False)
    for pattern in ("loss/*", "losses/*", "info/*"):
        run.define_metric(pattern, step_metric="global_step", step_sync=False)
    # W&B also registers referenced axes as metrics; make this axis's own chart rule explicit.
    run.define_metric("info/epochs", step_metric="global_step", step_sync=False)
