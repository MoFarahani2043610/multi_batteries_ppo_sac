import json, os

for algo in ['sac','ppo']:
    best_ret, best_seed = -999, -1
    for seed in range(5):
        path = f'm3_logs/{algo}_sweep/N1/seed_{seed}/cell_result.json'
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            print(f'  {algo} seed={seed}: return=${d["mean_return"]:.2f}')
            if d['mean_return'] > best_ret:
                best_ret = d['mean_return']
                best_seed = seed
    print(f'  --> BEST: {algo.upper()} seed={best_seed}  return=${best_ret:.2f}')
    print()