"""
backfill_results_log.py — one-time script to seed results_log.csv with the
runs we already completed on REAL CAISO data, before this logging system
existed.

Run this ONCE, from the project root (<repo>), after
dropping results_log.py next to it:

    python backfill_results_log.py

Every number below was copied from actual terminal output already pasted
into this conversation — nothing here is invented. Check the "notes" field
on each row against your own terminal history before trusting it; if
anything looks off, fix the row (or delete it and re-log it) rather than
editing results_log.csv by hand later without a record of why.

Runs deliberately NOT included (do not backfill, do not guess numbers for):
  - Architecture / depth experiments      -> never rerun on real data
  - Reward-coefficient experiments        -> never rerun on real data
  - M3 scalability sweep (N=2,5,10,20)    -> status unclear after the
                                              data correction; re-verify
                                              which cells, if any, actually
                                              ran on the REAL cache before
                                              logging them
  - M4 throughput/UTD sweep               -> not yet run on real data
  - M5 cross-market transfer              -> ERCOT/PJM prices are
                                              synthetic-calibrated, not
                                              real; log these later with
                                              notes saying so explicitly,
                                              don't mix them in silently
                                              with real-CAISO rows
"""

from results_log import log_run

REAL_CACHE = "cache/caiso_2024_2026_5min.parquet"  # the file, post-swap, that
                                                     # now holds real CAISO data

# ── M1 baselines (run_v2.py, 50 episodes, real data) ──────────────────────
log_run(
    algo="Random", n_batteries=1, seed=None,
    dataset_path=REAL_CACHE, config_path="run_v2.py",
    total_timesteps=None, eval_episodes=50,
    eval_mean_return=-7.03, eval_std_return=12.16,
    notes="M1 baseline, real CAISO data, official number",
)
log_run(
    algo="Threshold", n_batteries=1, seed=None,
    dataset_path=REAL_CACHE, config_path="run_v2.py",
    total_timesteps=None, eval_episodes=50,
    eval_mean_return=17.80, eval_std_return=19.39,
    notes="M1 baseline, real CAISO data, official number",
)
log_run(
    algo="Oracle_LP", n_batteries=1, seed=None,
    dataset_path=REAL_CACHE, config_path="run_v2.py",
    total_timesteps=None, eval_episodes=50,
    eval_mean_return=65.11, eval_std_return=26.41,
    notes="M1 perfect-foresight LP baseline, real CAISO data, official number",
)

# ── M2 official results (5 seeds each, real data) ─────────────────────────
log_run(
    algo="SAC", n_batteries=1, seed="0-4 (mean)",
    dataset_path=REAL_CACHE, config_path="m2_configs/sac_n1.yaml",
    total_timesteps=1_000_000, eval_episodes=10,
    eval_mean_return=28.04, eval_std_return=0.86,
    notes="M2 official SAC result, real CAISO data. FLAG: reported as 37.6% of LP "
          "baseline elsewhere but the LP oracle figure that % was computed against "
          "does not match the M1 Oracle(LP)=$65.11 logged above -- resolve before citing.",
)
log_run(
    algo="PPO", n_batteries=1, seed="0-4 (mean)",
    dataset_path=REAL_CACHE, config_path="m2_configs/ppo_n1.yaml",
    total_timesteps=1_000_000, eval_episodes=10,
    eval_mean_return=24.49, eval_std_return=2.84,
    notes="M2 official PPO result, real CAISO data, 32.1% of LP oracle",
)

# ── SAC N=1 per-seed final returns (real data, from plot_per_seed_diagnostics_v2.py) ──
_per_seed_final = {0: 23.7, 1: 27.2, 2: 44.4, 3: 29.5, 4: 34.1}
for seed, final_return in _per_seed_final.items():
    log_run(
        algo="SAC", n_batteries=1, seed=seed,
        dataset_path=REAL_CACHE, config_path="m2_configs/sac_n1.yaml",
        total_timesteps=1_000_000, eval_episodes=10,
        eval_mean_return=final_return, eval_std_return=None,
        notes="Per-seed final eval return, real CAISO data, from per-seed diagnostics plot",
    )

# ── Richer-observation experiment (real data, 5 seeds) ─────────────────────
_richer_obs = {
    0: (20.25, 18.15),
    1: (21.41, 18.16),
    2: (23.79, 22.16),
    3: (21.50, 18.86),
    4: (20.80, 17.32),
}
for seed, (mean_ret, std_ret) in _richer_obs.items():
    log_run(
        algo="SAC_richer_obs", n_batteries=1, seed=seed,
        dataset_path=REAL_CACHE,
        config_path=f"test_richer_observation_seed{seed}.py" if seed else "test_richer_observation.py",
        total_timesteps=1_000_000, eval_episodes=10,
        eval_mean_return=mean_ret, eval_std_return=std_ret,
        notes="Richer-observation ablation, real CAISO data. Result: WORSE than "
              "baseline SAC ($28.04) by ~23% -- reversed from the old synthetic-data finding.",
    )

# ── Train/test leakage verification (real data) ─────────────────────────
_leakage = {1: (29.22, 12.18), 2: (29.51, 12.35), 4: (29.54, 11.66)}
for seed, (mean_ret, std_ret) in _leakage.items():
    log_run(
        algo="SAC_traintest_split", n_batteries=1, seed=seed,
        dataset_path=REAL_CACHE,
        config_path=f"test_traintest_split_seed{seed}.py",
        total_timesteps=1_000_000, eval_episodes=10,
        eval_mean_return=mean_ret, eval_std_return=std_ret,
        notes="Train/eval leakage check (temporally-separated split), real CAISO data. "
              "Comparable to the M2 baseline ($28.04); a check on a different window, not an exact comparison.",
    )

print("Backfill complete. Review results_log.csv before citing any row in the thesis.")