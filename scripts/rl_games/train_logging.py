"""Training video upload and cleanup helpers, independent of the simulator."""

import sys
from contextlib import contextmanager
from pathlib import Path


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


def video_file_states(folder):
    """Identify new or overwritten MP4 files when a run directory is reused."""
    if folder is None:
        return {}
    return {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in Path(folder).glob("*.mp4")}


def upload_training_videos(run, folder, previous_states):
    """Upload finalized recordings explicitly, without W&B's Gym monkey patch."""
    if run is None or folder is None:
        return
    import wandb

    for path, state in sorted(video_file_states(folder).items()):
        if state != previous_states.get(path):
            run.log({f"videos/train/{path.stem}": wandb.Video(str(path), format="mp4")})
