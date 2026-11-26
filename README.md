# Scalability of Deep RL for Multi-Battery Energy Storage Arbitrage: SAC vs PPO

Master's thesis code (Sapienza, LM-29 Electronics Engineering), Mohaddeseh Farahani.
Environment: N-battery arbitrage on real CAISO 5-minute prices (TH_NP15_GEN-APND). Algorithms: SAC and PPO (Stable-Baselines3).

## Data
- Real CAISO LMP, 2024-09-01 to 2026-09-23, 216,288 five-minute steps (751 of 753 days; the days 26-27 Nov 2024 are missing).
- The verified dataset is included as `cache/caiso_2024_2026_5min.parquet` (SHA-256 starts with 06bae5ea). It can be re-created with `python download_full_real_caiso.py`. Older commits (before 40bf327) contain a synthetic dataset that must not be used.
- Split by index: train 192,095 steps (to 1 Jul 2026), validation 8,928 (2 Jul - 1 Aug), test 15,265 (2 Aug - 23 Sep 2026).
- Note: an earlier synthetic cache was used in some early experiments. It was replaced by the real data (commit 40bf327). All results in the thesis come from the real data. All configs use `price_ref: 35.28`.

## How the thesis results are produced
1. Train (one cell = algorithm, N, seed; seed = heterogeneous fleet + RL randomness):
   `python m3_scripts/scale_sweep.py --config m3_configs/ppo_lr1e4_scaling.yaml --algo ppo --n-values 1 2 5 10 20 --seeds 0 1 2 3 4`
   Configs: `m3_configs/{ppo,sac}_scaling.yaml` (learning rate 1e-3), `{ppo,sac}_lr1e4_scaling.yaml` (learning rate 1e-4). Other `sac_n5_*` and `sac_n10_*` configs are the screening experiments of Chapter 5.
2. Re-evaluate saved models on their own fleet (20 episodes): `python m3_reeval.py --algo ppo --root m3_logs\ppo_lr1e4_sweep --label ppo_lr1e4` -> `m3_reeval_episodes.csv`.
   IMPORTANT: the summary tables printed by `scale_sweep.py` are not valid for N>1 (they evaluate on identical default batteries). Only `m3_reeval.py` results are used.
3. Baselines (Threshold, LP) on the same fleets: `m3_baselines.py` -> `m3_baselines_episodes.csv`, `m3_baselines_summary.csv`.
4. Paired tests across fleets (Bonferroni 0.01): `m3_paired.py`, `m3_paired_lr1e4.py` -> `m3_paired_summary.csv`, `m3_paired_lr1e4_summary.csv`.
5. SAC divergence criterion (median critic loss over the last 100K steps > 100): `m3_tb_diag.py` -> `m3_tb_diag_runs.csv`.
6. N=1 diagnostics figures: `diag_sac_n1_real.py` -> `figs_real/`.

## Run log
`results_log.csv` records runs with dataset hash and config path (`results_log.py`).

## Not in the repository
Trained models, TensorBoard logs and the dataset (see `.gitignore`).

