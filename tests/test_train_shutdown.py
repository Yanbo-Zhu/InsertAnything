"""Exercise the training entry point with real Gymnasium video and simulated RL/Kit."""

import io
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import gymnasium as gym
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "rl_games"
sys.path.insert(0, str(SCRIPT_DIR))


class TrainingShutdownTests(unittest.TestCase):
    def execute(self, *, track=True, video=False, failure=None, rank=0, app_failure=False):
        events = []
        training_error = RuntimeError("training failed")
        close_error = RuntimeError("environment close failed")
        output = io.StringIO()
        modules = {}

        def module(name, **attrs):
            result = ModuleType(name)
            result.__path__ = []
            result.__dict__.update(attrs)
            modules[name] = result
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(modules[parent], child, result)
            return result

        class AppLauncher:
            @staticmethod
            def add_app_launcher_args(parser):
                parser.add_argument("--device", default="cpu")
                parser.add_argument("--enable_cameras", action="store_true")

            def __init__(self, args):
                self.app = SimpleNamespace(close=close_app)

        def close_app():
            events.append("app.close")
            if app_failure:
                raise RuntimeError("app close failed")

        class Env(gym.Env):
            metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
            render_mode = "rgb_array"
            num_envs = 4
            observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
            action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                return np.zeros(1, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

            def render(self):
                return np.zeros((16, 16, 3), dtype=np.uint8)

            def close(self):
                events.append("env.close")
                if failure == "close":
                    raise close_error

        class RlWrapper(gym.Wrapper):
            def __init__(self, env, *args):
                super().__init__(env)
                state.env = self

        state = SimpleNamespace(env=None, writer=None)
        env_cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=4), sim=SimpleNamespace(device="cpu"))
        env_cfg.to_dict = lambda: {"seed": env_cfg.seed}
        agent_cfg = {
            "params": {
                "seed": 0,
                "env": {},
                "config": {"name": "test", "full_experiment_name": "run", "max_epochs": 1, "device": "cpu"},
            }
        }

        def hydra_task_config(*args):
            return lambda func: lambda: func(env_cfg, agent_cfg)

        def train(args):
            if track and rank == 0:
                self.assertEqual(run.define_metric.call_count, 7)
            events.append("train")
            state.writer = SummaryWriter("summaries")
            state.writer.add_scalar("test/loss", 0.5, 512)
            runner.algo_observer.writer = SimpleNamespace(close=close_writer)
            state.env.reset()
            state.env.step(np.zeros(1, dtype=np.float32))
            if failure in ("train", "train_and_finish"):
                raise training_error

        def close_writer():
            events.append("writer.close")
            state.writer.close()
            if failure == "writer":
                raise RuntimeError("writer close failed")

        def finish(*, exit_code):
            events.append(f"wandb.finish:{exit_code}")
            if state.writer is not None:
                events_file = EventAccumulator("summaries").Reload()
                self.assertEqual(events_file.Scalars("test/loss")[0].value, 0.5)
            if failure in ("finish", "train_and_finish"):
                raise RuntimeError("W&B finish failed")

        run = SimpleNamespace(
            resumed=False, config=Mock(), define_metric=Mock(),
            log=Mock(side_effect=AssertionError("Training must sync scalars only through TensorBoard")),
            finish=Mock(side_effect=finish),
        )
        wandb = module(
            "wandb", init=Mock(return_value=run),
            Video=Mock(side_effect=AssertionError("Training videos must remain local")),
        )
        if failure == "init":
            wandb.init.side_effect = RuntimeError("W&B init failed")
        if failure == "axes":
            run.define_metric.side_effect = RuntimeError("W&B axis configuration failed")
        if not track or rank != 0:
            modules["wandb"] = None  # Any accidental W&B import fails in these modes.
        runner = Mock()
        runner.algo_observer = SimpleNamespace(writer=None)
        runner.run.side_effect = train
        module("isaaclab")
        module("isaaclab.app", AppLauncher=AppLauncher)
        module(
            "isaaclab.envs",
            DirectMARLEnv=type("MARLEnv", (), {}),
            DirectMARLEnvCfg=type("MARLCfg", (), {}),
            DirectRLEnvCfg=type("DirectCfg", (), {}),
            ManagerBasedRLEnvCfg=type("ManagerCfg", (), {}),
            multi_agent_to_single_agent=Mock(),
        )
        module("isaaclab.utils")
        module("isaaclab.utils.assets", retrieve_file_path=lambda path: path)
        module("isaaclab.utils.dict", print_dict=Mock())
        module("isaaclab.utils.io", dump_pickle=Mock(), dump_yaml=Mock())
        module("isaaclab_rl")
        module("isaaclab_rl.rl_games", RlGamesGpuEnv=Mock(), RlGamesVecEnvWrapper=RlWrapper)
        module("isaaclab_tasks")
        module("isaaclab_tasks.utils")
        module("isaaclab_tasks.utils.hydra", hydra_task_config=hydra_task_config)
        module("InsertAnything")
        module("InsertAnything.tasks")
        module("omni", log=SimpleNamespace(warn=Mock()))
        module("rl_games")
        module("rl_games.common", env_configurations=Mock(), vecenv=Mock())
        module("rl_games.torch_runner", Runner=Mock(return_value=runner))

        argv = [str(SCRIPT_DIR / "train.py"), "--task", "test"]
        if track:
            argv += ["--track", "--wandb-entity", "test"]
        if video:
            argv += ["--video", "--video_length", "10"]  # Leave a partial clip for close() to finalize.
        caught = None
        with tempfile.TemporaryDirectory() as folder:
            previous_cwd = os.getcwd()
            try:
                os.chdir(folder)
                with (
                    patch.dict(sys.modules, modules),
                    patch.object(sys, "argv", argv),
                    patch.dict(os.environ, {"RANK": str(rank)}),
                    patch.object(gym, "make", return_value=Env()),
                    redirect_stdout(output),
                    redirect_stderr(output),
                ):
                    try:
                        runpy.run_path(str(SCRIPT_DIR / "train.py"), run_name="__main__")
                    except BaseException as exc:
                        caught = exc
                clips = list(Path(folder).rglob("*.mp4"))
                clip_sizes = [path.stat().st_size for path in clips]
            finally:
                os.chdir(previous_cwd)
        return SimpleNamespace(
            events=events, error=caught, training_error=training_error,
            close_error=close_error, wandb=wandb, run=run, clip_sizes=clip_sizes,
        )

    def test_video_is_finalized_locally_without_upload_before_successful_shutdown(self):
        result = self.execute(video=True)
        self.assertIsNone(result.error)
        self.assertEqual(
            result.events, ["train", "env.close", "writer.close", "wandb.finish:0", "app.close"]
        )
        self.assertTrue(result.clip_sizes)
        self.assertFalse(result.wandb.init.call_args.kwargs["monitor_gym"])
        self.assertTrue(result.wandb.init.call_args.kwargs["sync_tensorboard"])
        result.wandb.Video.assert_not_called()
        result.run.log.assert_not_called()
        self.assertEqual(result.run.define_metric.call_args_list, [
            call(pattern, step_metric=axis, step_sync=False)
            for pattern, axis in (
                ("train/*", "info/epochs"), ("reward_terms_raw/*", "info/epochs"),
                ("performance/*", "info/epochs"), ("loss/*", "global_step"),
                ("losses/*", "global_step"), ("info/*", "global_step"),
                ("info/epochs", "global_step"),
            )
        ])

    def test_training_error_still_finalizes_partial_video_and_marks_run_failed(self):
        result = self.execute(video=True, failure="train")
        self.assertIs(result.error, result.training_error)
        self.assertEqual(
            result.events[-4:], ["env.close", "writer.close", "wandb.finish:1", "app.close"]
        )
        self.assertTrue(result.clip_sizes)
        result.wandb.Video.assert_not_called()
        result.run.log.assert_not_called()

    def test_environment_close_error_still_finishes_wandb_and_closes_app(self):
        result = self.execute(failure="close")
        self.assertIs(result.error, result.close_error)
        self.assertEqual(result.events[-2:], ["wandb.finish:1", "app.close"])

    def test_axis_configuration_error_still_finishes_wandb_and_closes_app(self):
        result = self.execute(failure="axes")
        self.assertEqual(str(result.error), "W&B axis configuration failed")
        self.assertEqual(result.events, ["env.close", "wandb.finish:1", "app.close"])

    def test_writer_close_error_still_finishes_wandb_and_closes_app(self):
        result = self.execute(failure="writer")
        self.assertEqual(str(result.error), "writer close failed")
        self.assertEqual(result.events[-2:], ["wandb.finish:1", "app.close"])

    def test_finish_error_still_closes_app(self):
        result = self.execute(failure="finish")
        self.assertEqual(str(result.error), "W&B finish failed")
        self.assertEqual(result.events[-1], "app.close")

    def test_cleanup_errors_do_not_replace_training_error(self):
        result = self.execute(failure="train_and_finish", app_failure=True)
        self.assertIs(result.error, result.training_error)
        self.assertIn("W&B finish failed", " ".join(result.error.__notes__))
        self.assertIn("app close failed", " ".join(result.error.__notes__))
        self.assertEqual(result.events[-1], "app.close")

    def test_init_failure_still_closes_environment_and_app(self):
        result = self.execute(failure="init")
        self.assertEqual(str(result.error), "W&B init failed")
        self.assertEqual(result.events, ["env.close", "app.close"])

    def test_local_video_does_not_import_wandb(self):
        result = self.execute(track=False, video=True)
        self.assertIsNone(result.error)
        self.assertTrue(result.clip_sizes)
        self.assertEqual(result.events, ["train", "env.close", "writer.close", "app.close"])

    def test_tracking_without_video_does_not_upload_video(self):
        result = self.execute()
        self.assertIsNone(result.error)
        self.assertEqual(result.events, ["train", "env.close", "writer.close", "wandb.finish:0", "app.close"])

    def test_secondary_rank_does_not_import_wandb(self):
        result = self.execute(rank=1)
        self.assertIsNone(result.error)
        self.assertEqual(result.events, ["train", "env.close", "writer.close", "app.close"])


if __name__ == "__main__":
    unittest.main()
