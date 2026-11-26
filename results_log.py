"""
results_log.py — append-only experiment ledger for the thesis.

Every training/eval run appends ONE row to results_log.csv. Nothing is ever
edited or deleted in place; if a run is redone, it gets a new row. This is
what you show Danial (or put in an appendix table) as evidence of what was
actually run, on what data, with what config, and what came out.

Usage (wrap around an existing training script):

    from results_log import RunLogger

    with RunLogger(
        algo="SAC",
        n_batteries=1,
        seed=0,
        dataset_path="cache/caiso_2024_2026_5min.parquet",
        config_path="m2_configs/sac_n1.yaml",
        total_timesteps=1_000_000,
        eval_episodes=10,
        notes="M2 hyperparameter run, real CAISO data",
    ) as run:
        model.learn(total_timesteps=1_000_000)
        mean_ret, std_ret = evaluate(model, env, n_episodes=10)
        run.record_result(mean_ret, std_ret)

If the `with` block raises an exception, the row is still written with
status=FAILED and the error message captured — you never lose the record
that a run was attempted and crashed.

You can also log a row by hand (e.g. for a run you already finished before
this system existed):

    from results_log import log_run
    log_run(
        algo="SAC", n_batteries=1, seed=0,
        dataset_path="cache/caiso_2024_2026_5min.parquet",
        config_path="m2_configs/sac_n1.yaml",
        total_timesteps=1_000_000, eval_episodes=10,
        eval_mean_return=28.04, eval_std_return=0.86,
        runtime_minutes=None, status="COMPLETED",
        notes="Backfilled from terminal output, real CAISO data (M2 official)",
    )
"""

import contextlib
import csv
import hashlib
import os
import time
import traceback
from datetime import datetime, timezone

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_log.csv")

FIELDNAMES = [
    "run_id",
    "timestamp_utc",
    "algo",
    "n_batteries",
    "seed",
    "dataset_path",
    "dataset_sha256_8",   # first 8 hex chars of the file's sha256 — enough to
                           # prove "same file" or "different file" without a
                           # huge column; catches the exact fake/real-swap bug
                           # that happened this project
    "config_path",
    "total_timesteps",
    "eval_episodes",
    "eval_mean_return",
    "eval_std_return",
    "runtime_minutes",
    "status",              # COMPLETED / FAILED / RUNNING
    "error",
    "notes",
]


LOCK_PATH = CSV_PATH + ".lock"


@contextlib.contextmanager
def _file_lock(timeout=60.0, poll=0.1, stale_after=120.0):
    """Cross-process lock via an exclusively-created lock file, so two
    training terminals can log at the same time without ever getting the
    same run_id or interleaving a row. A lock older than stale_after seconds
    is treated as left over from a crashed process and removed."""
    start = time.time()
    fd = None
    while True:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK_PATH) > stale_after:
                    os.remove(LOCK_PATH)
                    continue
            except OSError:
                pass
            if time.time() - start > timeout:
                raise TimeoutError(f"Could not acquire {LOCK_PATH} within {timeout}s")
            time.sleep(poll)
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(LOCK_PATH)
        except OSError:
            pass


def _file_hash8(path):
    """First 8 hex chars of sha256(file). Returns 'MISSING' if not found."""
    if not path or not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _next_run_id():
    if not os.path.exists(CSV_PATH):
        return 1
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 1
    return max(int(r["run_id"]) for r in rows) + 1


def _ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def log_run(
    algo,
    n_batteries,
    seed,
    dataset_path,
    config_path,
    total_timesteps,
    eval_episodes,
    eval_mean_return=None,
    eval_std_return=None,
    runtime_minutes=None,
    status="COMPLETED",
    error="",
    notes="",
):
    """Append one row. Safe to call directly for backfilling past runs."""
    row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "algo": algo,
        "n_batteries": n_batteries,
        "seed": seed,
        "dataset_path": dataset_path,
        "dataset_sha256_8": _file_hash8(dataset_path),
        "config_path": config_path,
        "total_timesteps": total_timesteps,
        "eval_episodes": eval_episodes,
        "eval_mean_return": "" if eval_mean_return is None else round(eval_mean_return, 4),
        "eval_std_return": "" if eval_std_return is None else round(eval_std_return, 4),
        "runtime_minutes": "" if runtime_minutes is None else round(runtime_minutes, 2),
        "status": status,
        "error": error,
        "notes": notes,
    }
    # run_id is assigned and the row written inside the lock, so concurrent
    # processes can never receive the same id.
    with _file_lock():
        _ensure_csv()
        row["run_id"] = _next_run_id()
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
    return row["run_id"]


class RunLogger:
    """Context manager: times the run, catches exceptions, writes one row."""

    def __init__(
        self,
        algo,
        n_batteries,
        seed,
        dataset_path,
        config_path,
        total_timesteps,
        eval_episodes,
        notes="",
    ):
        self.algo = algo
        self.n_batteries = n_batteries
        self.seed = seed
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.total_timesteps = total_timesteps
        self.eval_episodes = eval_episodes
        self.notes = notes
        self._mean = None
        self._std = None
        self._t0 = None

    def record_result(self, mean_return, std_return):
        self._mean = mean_return
        self._std = std_return

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        runtime_minutes = (time.time() - self._t0) / 60.0
        if exc_type is None:
            log_run(
                algo=self.algo,
                n_batteries=self.n_batteries,
                seed=self.seed,
                dataset_path=self.dataset_path,
                config_path=self.config_path,
                total_timesteps=self.total_timesteps,
                eval_episodes=self.eval_episodes,
                eval_mean_return=self._mean,
                eval_std_return=self._std,
                runtime_minutes=runtime_minutes,
                status="COMPLETED" if self._mean is not None else "RUNNING",
                notes=self.notes,
            )
        else:
            err = "".join(traceback.format_exception_only(exc_type, exc_val)).strip()
            log_run(
                algo=self.algo,
                n_batteries=self.n_batteries,
                seed=self.seed,
                dataset_path=self.dataset_path,
                config_path=self.config_path,
                total_timesteps=self.total_timesteps,
                eval_episodes=self.eval_episodes,
                eval_mean_return=self._mean,
                eval_std_return=self._std,
                runtime_minutes=runtime_minutes,
                status="FAILED",
                error=err,
                notes=self.notes,
            )
        # Don't suppress the exception — the caller/terminal should still see it.
        return False