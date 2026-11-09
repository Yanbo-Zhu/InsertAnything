"""Deterministic evaluation logging tests; no simulator or GPU is needed."""

import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rl_games"))
from play_logging import EvaluationMetrics, PlayLogger


def snapshot(successes, time_sum=0.0):
    return {"success_count": successes, "episode_count": 2, "success_time_sum_s": time_sum}


class EvaluationMetricsTests(unittest.TestCase):
    def test_rounds_cumulative_statistics_and_reward_means(self):
        accumulator = EvaluationMetrics()
        self.assertIsNone(accumulator.step(torch.tensor([1.0, 3.0]), {"logs_rew_align": torch.tensor(2.0)}))
        first = accumulator.step(
            torch.tensor([3.0, 5.0]), {"logs_rew_align": torch.tensor(6.0), "eval_metrics": snapshot(2, 4.0)}
        )
        self.assertEqual(first["eval/mean_return"], 6.0)
        self.assertEqual(first["reward_terms_raw/align"], 4.0)
        self.assertEqual(first["eval/mean_success_time_s"], 2.0)
        self.assertEqual(first["eval/success_rate_percent"], 100.0)
        zero = accumulator.step(torch.tensor([10.0, 20.0]), {"eval_metrics": snapshot(0)})
        self.assertEqual(zero["eval/mean_return"], 15.0)
        self.assertEqual(zero["eval/success_rate_percent"], 0.0)
        self.assertNotIn("eval/mean_success_time_s", zero)
        self.assertNotIn("reward_terms_raw/align", zero)
        last = accumulator.step(torch.zeros(2), {"eval_metrics": snapshot(1, 18.0)})
        self.assertEqual(last["progress/round"], 3)
        self.assertEqual(last["progress/control_steps"], 4)
        self.assertEqual(last["progress/environment_episodes"], 6)
        self.assertEqual(last["eval/cumulative_success_rate_percent"], 50.0)
        self.assertAlmostEqual(last["eval/cumulative_mean_success_time_s"], 22.0 / 3.0)

    def test_no_successes_never_writes_success_times(self):
        metrics = EvaluationMetrics().step(torch.zeros(2), {"eval_metrics": snapshot(0)})
        self.assertNotIn("eval/mean_success_time_s", metrics)
        self.assertNotIn("eval/cumulative_mean_success_time_s", metrics)

    def test_success_at_time_zero_is_retained(self):
        metrics = EvaluationMetrics().step(torch.zeros(2), {"eval_metrics": snapshot(1)})
        self.assertEqual(metrics["eval/mean_success_time_s"], 0.0)


class PlayLoggerTests(unittest.TestCase):
    def make_wandb(self):
        run = Mock()
        run.offline = False
        run.settings.mode = "online"
        run.url = "https://wandb.example/test"
        return SimpleNamespace(init=Mock(return_value=run), Settings=Mock()), run

    def test_both_backends_receive_identical_values_and_steps(self):
        wandb, run = self.make_wandb()
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {"wandb": wandb}):
            logger = PlayLogger(root, "test", {"seed": 0}, track=True)
            metrics = EvaluationMetrics().step(torch.tensor([1.0, 3.0]), {"eval_metrics": snapshot(1, 5.0)})
            logger.log(metrics)
            logger.close()
            logger.close()  # Cleanup is safe after partial initialization or a previous close.
            run.log.assert_called_once_with(metrics, step=1)
            run.finish.assert_called_once_with(exit_code=0)
            self.assertFalse(wandb.init.call_args.kwargs["sync_tensorboard"])
            events = EventAccumulator(str(logger.directory / "tensorboard")).Reload()
            self.assertEqual(set(events.Tags()["scalars"]), set(metrics))
            for name, value in metrics.items():
                event = events.Scalars(name)[0]
                self.assertEqual(event.step, 1)
                self.assertAlmostEqual(event.value, value, places=5)

    def test_local_only_never_imports_wandb_and_directories_are_unique(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("play_logging.importlib.import_module", wraps=importlib.import_module) as imports:
                first = PlayLogger(root, "same/name", {})
                first.close()
                second = PlayLogger(root, "same/name", {})
                second.close()
                self.assertFalse(any(call.args[0] == "wandb" for call in imports.call_args_list))
            self.assertNotEqual(first.directory, second.directory)
            self.assertEqual(first.directory.parent, Path(root))
            self.assertTrue(list(first.directory.rglob("events.out.tfevents.*")))
            self.assertTrue(list(second.directory.rglob("events.out.tfevents.*")))

    def test_wandb_init_failure_is_fatal_and_tensorboard_is_closed(self):
        wandb, _ = self.make_wandb()
        wandb.init.side_effect = RuntimeError("permission denied")
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {"wandb": wandb}):
            with self.assertRaisesRegex(RuntimeError, "permission denied"):
                PlayLogger(root, "failure", {}, track=True)
            directory = next(Path(root).iterdir())
            events = EventAccumulator(str(directory / "tensorboard")).Reload()
            self.assertIn("run/config/text_summary", events.Tags()["tensors"])

    def test_wandb_write_failure_is_fatal_and_local_metrics_survive(self):
        wandb, run = self.make_wandb()
        run.log.side_effect = RuntimeError("connection failed")
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {"wandb": wandb}):
            logger = PlayLogger(root, "failure", {}, track=True)
            with self.assertRaisesRegex(RuntimeError, "W&B metric logging failed"):
                logger.log({"progress/round": 1, "eval/success_rate_percent": 50.0})
            logger.close(exit_code=1)
            run.finish.assert_called_once_with(exit_code=1)
            events = EventAccumulator(str(logger.directory / "tensorboard")).Reload()
            self.assertEqual(events.Scalars("eval/success_rate_percent")[0].value, 50.0)

    def test_disabled_wandb_is_not_silently_accepted(self):
        wandb, run = self.make_wandb()
        run.settings.mode = "disabled"
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {"wandb": wandb}):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                PlayLogger(root, "disabled", {}, track=True)
            run.finish.assert_called_once_with(exit_code=1)


if __name__ == "__main__":
    unittest.main()
