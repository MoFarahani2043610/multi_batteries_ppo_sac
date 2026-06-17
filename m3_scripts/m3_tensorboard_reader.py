"""
m3_scripts/m3_tensorboard_reader.py
=====================================
Extracts action entropy and ent_coef from TensorBoard event files
for all M3 training cells.

Metrics extracted:
  - SAC: train/ent_coef (entropy temperature across training)
  - PPO: train/entropy_loss (entropy loss across training)
  - Both: rollout/ep_rew_mean (for sample efficiency calculation)

Output:
  - experiments/m3_action_dim/entropy_data.json
  - experiments/m3_action_dim/learning_curves.json

Usage
-----
    python m3_scripts/m3_tensorboard_reader.py

Requires: tensorboard (pip install tensorboard)
"""

from __future__ import annotations
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _ROOT)

N_VALUES = [1, 2, 5, 10, 20]
ALGOS    = ['sac', 'ppo']
SEEDS    = list(range(5))
LOG_BASE = os.path.join(_ROOT, 'm3_logs')
OUT_DIR  = os.path.join(_ROOT, 'experiments', 'm3_action_dim')
os.makedirs(OUT_DIR, exist_ok=True)


def find_event_file(algo: str, n: int, seed: int) -> str | None:
    """Find the TensorBoard event file for a given cell."""
    sweep   = 'sac_sweep' if algo == 'sac' else 'ppo_sweep'
    seed_dir = os.path.join(LOG_BASE, sweep, f'N{n}', f'seed_{seed}')
    if not os.path.exists(seed_dir):
        return None
    # TensorBoard logs are in subdirectories named e.g. SAC_seed0_1/
    for item in os.listdir(seed_dir):
        subdir = os.path.join(seed_dir, item)
        if os.path.isdir(subdir):
            for fname in os.listdir(subdir):
                if fname.startswith('events.out.tfevents'):
                    return os.path.join(subdir, fname)
    return None


def read_scalar(event_file: str, tag: str) -> list[tuple[int, float]]:
    """
    Read a scalar tag from a TensorBoard event file.
    Returns list of (step, value) pairs.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(os.path.dirname(event_file))
        ea.Reload()
        if tag not in ea.Tags().get('scalars', []):
            return []
        events = ea.Scalars(tag)
        return [(e.step, e.value) for e in events]
    except Exception as e:
        print(f'    Warning: could not read {tag} from {event_file}: {e}')
        return []


def extract_all_entropy_data() -> dict:
    """Extract entropy metrics for all cells."""
    entropy_data = {}

    for algo in ALGOS:
        entropy_data[algo] = {}
        tag = 'train/ent_coef' if algo == 'sac' else 'train/entropy_loss'

        for n in N_VALUES:
            entropy_data[algo][n] = {}

            for seed in SEEDS:
                event_file = find_event_file(algo, n, seed)
                if event_file is None:
                    print(f'  {algo.upper()} N={n} seed={seed}: no event file found')
                    continue

                print(f'  Reading {algo.upper()} N={n} seed={seed}...')
                values = read_scalar(event_file, tag)

                if not values:
                    print(f'    No {tag} data found')
                    continue

                steps   = [v[0] for v in values]
                scalars = [v[1] for v in values]

                entropy_data[algo][n][seed] = {
                    'tag':        tag,
                    'steps':      steps,
                    'values':     scalars,
                    'final':      scalars[-1] if scalars else None,
                    'mean':       float(np.mean(scalars)) if scalars else None,
                    'min':        float(np.min(scalars))  if scalars else None,
                    'max':        float(np.max(scalars))  if scalars else None,
                }

    return entropy_data


def extract_learning_curves() -> dict:
    """Extract ep_rew_mean for sample efficiency calculation."""
    curves = {}

    for algo in ALGOS:
        curves[algo] = {}
        tag = 'rollout/ep_rew_mean'

        for n in N_VALUES:
            curves[algo][n] = {}

            for seed in SEEDS:
                event_file = find_event_file(algo, n, seed)
                if event_file is None:
                    continue

                values = read_scalar(event_file, tag)
                if not values:
                    continue

                steps   = [v[0] for v in values]
                returns = [v[1] for v in values]

                # compute steps to 80% of final return
                final_return = returns[-1] if returns else 0
                target_80    = 0.80 * final_return
                steps_to_80  = None
                for s, r in zip(steps, returns):
                    if r >= target_80:
                        steps_to_80 = s
                        break

                curves[algo][n][seed] = {
                    'steps':        steps,
                    'returns':      returns,
                    'final_return': final_return,
                    'steps_to_80':  steps_to_80,
                }

    return curves


def summarise_entropy(entropy_data: dict) -> None:
    """Print entropy summary table."""
    print('\n' + '='*70)
    print('  ENTROPY SUMMARY (final ent_coef for SAC, entropy_loss for PPO)')
    print('='*70)
    print(f"  {'N':>4}  {'SAC final ent_coef':>22}  {'PPO entropy_loss':>20}")
    print('  ' + '-'*50)

    for n in N_VALUES:
        sac_vals, ppo_vals = [], []

        for seed in SEEDS:
            sd = entropy_data['sac'].get(n, {}).get(seed)
            pd = entropy_data['ppo'].get(n, {}).get(seed)
            if sd and sd['final'] is not None:
                sac_vals.append(sd['final'])
            if pd and pd['final'] is not None:
                ppo_vals.append(abs(pd['final']))  # entropy_loss is negative

        sac_str = f"{np.mean(sac_vals):.4f}±{np.std(sac_vals):.4f}" if sac_vals else "pending"
        ppo_str = f"{np.mean(ppo_vals):.4f}±{np.std(ppo_vals):.4f}" if ppo_vals else "pending"
        print(f"  {n:>4}  {sac_str:>22}  {ppo_str:>20}")

    print('='*70)


def summarise_sample_efficiency(curves: dict) -> None:
    """Print sample efficiency summary."""
    print('\n' + '='*70)
    print('  SAMPLE EFFICIENCY (steps to 80% of final return)')
    print('='*70)
    print(f"  {'N':>4}  {'SAC steps (K)':>18}  {'PPO steps (K)':>18}")
    print('  ' + '-'*44)

    for n in N_VALUES:
        sac_vals, ppo_vals = [], []

        for seed in SEEDS:
            sc = curves['sac'].get(n, {}).get(seed)
            pc = curves['ppo'].get(n, {}).get(seed)
            if sc and sc['steps_to_80'] is not None:
                sac_vals.append(sc['steps_to_80'] / 1000)
            if pc and pc['steps_to_80'] is not None:
                ppo_vals.append(pc['steps_to_80'] / 1000)

        sac_str = f"{np.mean(sac_vals):.0f}K±{np.std(sac_vals):.0f}K" if sac_vals else "pending"
        ppo_str = f"{np.mean(ppo_vals):.0f}K±{np.std(ppo_vals):.0f}K" if ppo_vals else "pending"
        print(f"  {n:>4}  {sac_str:>18}  {ppo_str:>18}")

    print('='*70)


if __name__ == '__main__':
    print('\n' + '='*60)
    print('  M3 TensorBoard Reader')
    print('='*60)

    print('\nExtracting entropy data...')
    entropy_data = extract_all_entropy_data()

    print('\nExtracting learning curves...')
    curves = extract_learning_curves()

    # save JSON
    entropy_path = os.path.join(OUT_DIR, 'entropy_data.json')
    curves_path  = os.path.join(OUT_DIR, 'learning_curves.json')

    # convert to serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    with open(entropy_path, 'w') as f:
        json.dump(make_serializable(entropy_data), f, indent=2)
    print(f'\nSaved: {entropy_path}')

    with open(curves_path, 'w') as f:
        json.dump(make_serializable(curves), f, indent=2)
    print(f'Saved: {curves_path}')

    summarise_entropy(entropy_data)
    summarise_sample_efficiency(curves)

    print('\nDone! Use these JSONs in m3_analysis.py for complete metrics.')