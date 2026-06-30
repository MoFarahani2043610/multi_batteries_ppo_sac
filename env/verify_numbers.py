import json, os, numpy as np

print("="*80)
print("VERIFICATION OF ALL M3 NUMBERS")
print("="*80)

for algo in ['sac', 'ppo']:
    print(f"\n{algo.upper()}:")
    print(f"  {'N':>3}  {'Mean Ret':>10}  {'Std':>10}  {'LP%':>8}  {'Mean LP':>10}")
    print("  " + "-"*55)
    for n in [1,2,5,10,20]:
        returns, lps, fracs = [], [], []
        for seed in range(5):
            path = f'm3_logs/{algo}_sweep/N{n}/seed_{seed}/cell_result.json'
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                returns.append(d['mean_return'])
                lps.append(d['lp_return'])
                fracs.append(d['lp_fraction'])
        if returns:
            print(f"  {n:>3}  ${np.mean(returns):>9.2f}  ±${np.std(returns):>8.2f}  {np.mean(fracs)*100:>7.1f}%  ${np.mean(lps):>9.2f}")