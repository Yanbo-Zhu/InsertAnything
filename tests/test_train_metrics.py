"""Training metric arithmetic, reward bookkeeping and observer compatibility tests."""

import ast
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/rl_games"))
from train_logging import TrainingMetrics, TrainingMetricsObserver


def completed(successes, episodes, seconds=0.0, **terms):
    return {
        "train_episode_metrics": {
            "success_count": successes, "episode_count": episodes, "success_time_sum_s": seconds,
        },
        **terms,
    }


class TrainingMetricsTests(unittest.TestCase):
    def test_epoch_and_cumulative_statistics_use_episode_counts(self):
        metrics = TrainingMetrics()
        metrics.step({"logs_rew_align": torch.tensor(2.0)})
        first = metrics.end_epoch(1)
        self.assertEqual(first["train/episode_count"], 0)
        self.assertNotIn("train/success_rate_percent", first)
        self.assertNotIn("train/mean_return", first)
        self.assertEqual(first["reward_terms_raw/align"], 2.0)

        metrics.step(completed(1, 2, 4.0, logs_rew_align=torch.tensor(4.0)), torch.tensor([[5.0], [7.0]]))
        metrics.step(completed(1, 1, 2.0, logs_rew_align=torch.tensor(8.0)), torch.tensor([[9.0]]))
        second = metrics.end_epoch(2)
        self.assertEqual(second["train/success_count"], 2)
        self.assertEqual(second["train/episode_count"], 3)
        self.assertAlmostEqual(second["train/success_rate_percent"], 200.0 / 3.0)
        self.assertEqual(second["train/mean_return"], 7.0)
        self.assertEqual(second["train/mean_success_time_s"], 3.0)
        self.assertEqual(second["reward_terms_raw/align"], 6.0)
        self.assertEqual(second["progress/control_steps"], 3)
        self.assertEqual(second["progress/environment_episodes"], 3)

        metrics.step(completed(0, 1), torch.tensor([[10.0]]))
        third = metrics.end_epoch(3)
        self.assertEqual(third["train/success_rate_percent"], 0.0)
        self.assertEqual(third["train/cumulative_success_rate_percent"], 50.0)
        self.assertEqual(third["train/cumulative_success_count"], 2)
        self.assertEqual(third["train/cumulative_mean_success_time_s"], 3.0)
        self.assertNotIn("train/mean_success_time_s", third)
        self.assertNotIn("reward_terms_raw/align", third)

    def test_success_at_time_zero_is_valid(self):
        metrics = TrainingMetrics()
        metrics.step(completed(1, 1), torch.tensor([1.0]))
        values = metrics.end_epoch(1)
        self.assertEqual(values["train/mean_success_time_s"], 0.0)
        self.assertEqual(values["train/cumulative_mean_success_time_s"], 0.0)

    def test_reward_accumulation_does_not_modify_original_extras(self):
        term = torch.tensor(2.0)
        infos = {"logs_rew_align": term}
        metrics = TrainingMetrics()
        metrics.step(infos)
        metrics.step(infos)
        self.assertEqual(term.item(), 2.0)
        self.assertEqual(metrics.end_epoch(1)["reward_terms_raw/align"], 2.0)


class ObserverTests(unittest.TestCase):
    def test_original_callbacks_and_metrics_are_retained_without_duplicate_new_points(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = SummaryWriter(directory)
            original = Mock()
            original.after_print_stats.side_effect = lambda frame, epoch, seconds: writer.add_scalar(
                "scores/original", 0.25, frame
            )
            observer = TrainingMetricsObserver(original)
            algo = SimpleNamespace(writer=writer, current_rewards=torch.zeros((2, 1)))
            observer.before_init("run", {}, "test")
            observer.after_init(algo)
            observer.after_steps()
            observer.after_clear_stats()

            # An episode begins in epoch 1 and completes in epoch 2, including its terminal reward.
            algo.current_rewards[:] = torch.tensor([[2.0], [3.0]])
            observer.process_infos({"logs_rew_align": torch.tensor(1.0)}, torch.empty((0, 1), dtype=torch.long))
            writer.add_scalar("losses/a_loss", 0.75, 0)
            observer.after_print_stats(0, 1, 1.0)

            algo.current_rewards += torch.tensor([[4.0], [5.0]])
            infos = completed(1, 2, 1.5, logs_rew_align=torch.tensor(3.0))
            indices = torch.tensor([[0], [1]])
            observer.process_infos(infos, indices)
            # RL-Games clears completed episode buffers immediately after the observer returns.
            algo.current_rewards.zero_()
            writer.add_scalar("losses/a_loss", 0.5, 4)
            observer.after_print_stats(4, 2, 2.0)
            observer.after_print_stats(4, 2, 2.0)
            writer.close()

            original.before_init.assert_called_once_with("run", {}, "test")
            original.after_init.assert_called_once_with(algo)
            original.after_steps.assert_called_once()
            original.after_clear_stats.assert_called_once()
            self.assertEqual(original.process_infos.call_count, 2)
            self.assertIs(original.process_infos.call_args.args[0], infos)
            self.assertIs(original.process_infos.call_args.args[1], indices)
            self.assertEqual(original.after_print_stats.call_count, 3)

            events = EventAccumulator(directory).Reload()
            losses = events.Scalars("losses/a_loss")
            self.assertEqual([(event.step, event.value) for event in losses], [(0, 0.75), (4, 0.5)])
            self.assertEqual(len(events.Scalars("scores/original")), 3)
            self.assertEqual(len(events.Scalars("train/epoch")), 2)
            self.assertEqual(events.Scalars("train/mean_return")[0].value, 7.0)
            self.assertEqual(events.Scalars("train/success_rate_percent")[0].value, 50.0)
            self.assertEqual(events.Scalars("progress/environment_episodes")[-1].value, 2)
            tags = events.Tags()["scalars"]
            self.assertFalse(any(tag.startswith(("eval/", "train/reward_terms_raw/", "train/progress/")) for tag in tags))
            self.assertIn("reward_terms_raw/align", tags)

    def test_worker_without_writer_still_calls_original_observer(self):
        original = Mock()
        observer = TrainingMetricsObserver(original)
        observer.after_init(SimpleNamespace(writer=None))
        observer.process_infos(completed(1, 1), torch.tensor([[0]]))
        observer.after_print_stats(0, 1, 0.0)
        original.process_infos.assert_called_once()
        original.after_print_stats.assert_called_once()


class EnvironmentBookkeepingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the actual bookkeeping methods with tensor fixtures; importing the environment
        # normally requires starting Kit. Real train.py smoke tests cover the simulator integration.
        source_path = ROOT / "source/InsertAnything/InsertAnything/tasks/direct/insertanything/insertanything_env.py"
        tree = ast.parse(source_path.read_text())
        env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InsertAnythingEnv")
        methods = [node for node in env_class.body if isinstance(node, ast.FunctionDef) and node.name in {
            "_get_rewards", "_log_episode_metrics", "_reset_buffers",
        }]
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source_path), "exec"), namespace)
        cls.methods = namespace

    def make_env(self, evaluation=False):
        env = SimpleNamespace(
            cfg=SimpleNamespace(evaluation_mode=evaluation), device="cpu", num_envs=3, step_dt=0.1,
            ep_succeeded=torch.zeros(3, dtype=torch.long), ep_success_times=torch.zeros(3, dtype=torch.long),
            episode_length_buf=torch.zeros(3, dtype=torch.long), reset_buf=torch.zeros(3, dtype=torch.bool),
            actions=torch.zeros((3, 6)), extras={}, eval_ep_count=0,
        )
        for name in ("_get_rewards", "_log_episode_metrics", "_reset_buffers"):
            setattr(env, name, MethodType(self.methods[name], env))
        env.success_mask = torch.zeros(3, dtype=torch.bool)
        env._get_reward_terms = lambda: (
            {"curr_success": env.success_mask.float(), "align": torch.full((3,), 2.0)},
            {"curr_success": 3.0, "align": 0.5}, env.success_mask,
        )
        return env

    def test_latched_success_partial_resets_seconds_and_stale_payload(self):
        env = self.make_env()
        env.success_mask[:] = torch.tensor([True, False, True])
        env.episode_length_buf[:] = torch.tensor([2, 2, 5])
        self.assertEqual(env._get_rewards().tolist(), [4.0, 1.0, 4.0])
        self.assertNotIn("train_episode_metrics", env.extras)

        # Environment 0 has left the success region. Environment 2 has succeeded but is still running.
        env.success_mask.zero_()
        env.episode_length_buf += 1
        env.reset_buf[:] = torch.tensor([True, True, False])
        self.assertEqual(env._get_rewards().tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(env.extras["train_episode_metrics"], {
            "success_count": 1, "episode_count": 2, "success_time_sum_s": 0.2,
        })
        self.assertEqual(env.extras["successes"].item(), 0.0)  # Existing instantaneous statistic is retained.
        self.assertIn("logs_rew_align", env.extras)
        self.assertIn("success_times", env.extras)

        env._reset_buffers(torch.tensor([0, 1]))
        self.assertEqual(env.ep_succeeded.tolist(), [0, 0, 1])
        env.reset_buf.zero_()
        env._get_rewards()
        self.assertNotIn("train_episode_metrics", env.extras)
        env.reset_buf[2] = True
        env._get_rewards()
        self.assertEqual(env.extras["train_episode_metrics"]["success_count"], 1)
        self.assertEqual(env.extras["train_episode_metrics"]["success_time_sum_s"], 0.5)

    def test_terminal_step_success_is_included(self):
        env = self.make_env()
        env.success_mask[1] = True
        env.reset_buf[1] = True
        env.episode_length_buf[:] = 7
        env._get_rewards()
        summary = env.extras["train_episode_metrics"]
        self.assertEqual(summary["success_count"], 1)
        self.assertAlmostEqual(summary["success_time_sum_s"], 0.7)

    def test_evaluation_payload_and_printout_are_unchanged(self):
        env = self.make_env(evaluation=True)
        env.success_mask[0] = True
        env.reset_buf[:] = True
        env.episode_length_buf[:] = 3
        env._get_rewards()
        self.assertNotIn("train_episode_metrics", env.extras)
        self.assertEqual(env.extras["eval_metrics"]["success_count"], 1)
        self.assertEqual(env.extras["eval_metrics"]["episode_count"], 3)
        self.assertIn("33.33% (1/3)", env.extras["eval_printout"])


if __name__ == "__main__":
    unittest.main()
