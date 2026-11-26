import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = r"m2_logs\sac_n1_v2\N1"
OUT = "figs_real"; os.makedirs(OUT, exist_ok=True)
seeds = [0,1,2,3,4]
data = {}
for s in seeds:
    d = {}
    tbdir = os.path.join(ROOT, f"SAC_seed{s}_1")
    ea = EventAccumulator(tbdir, size_guidance={"scalars": 0}); ea.Reload()
    tags = ea.Tags()["scalars"]
    if s == 0: print("TB tags:", tags)
    for key, pat in [("actor","actor_loss"),("critic","critic_loss"),("alpha","ent_coef")]:
        t = [x for x in tags if x.endswith(pat)]
        if t:
            ev = ea.Scalars(t[0]); d[key] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    z = np.load(os.path.join(ROOT, f"seed_{s}", "eval_results", "evaluations.npz"))
    d["eval"] = (z["timesteps"], z["results"].mean(axis=1))
    data[s] = d

print("\nPER-SEED CHECK (for Section 4.1.1)")
for s in seeds:
    d = data[s]; st, ev = d["eval"]
    line = f"seed {s}: eval reward min {ev.min():.1f} max {ev.max():.1f} (max at step {st[ev.argmax()]:,})"
    if "critic" in d:
        cs, cv = d["critic"]; line += f" | critic max {cv.max():.0f} at step {cs[cv.argmax()]:,}, median {np.median(cv):.2f}"
    if "actor" in d:
        as_, av = d["actor"]; line += f" | actor min {av.min():.1f} max {av.max():.1f} (max at step {as_[av.argmax()]:,})"
    print(line)

cols = [("eval","Evaluation reward ($)"),("actor","Actor loss"),("critic","Critic loss"),("alpha","Entropy coefficient (alpha)")]
colors = plt.cm.tab10.colors
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for j, (k, title) in enumerate(cols):
    for s in seeds:
        if k in data[s]:
            x, y = data[s][k]
            if k != "eval" and len(y) > 2000:
                w = len(y)//500; y = np.convolve(y, np.ones(w)/w, mode="valid"); x = x[:len(y)]
            ax[j].plot(x, y, lw=0.8, color=colors[s], label=f"seed {s}")
    ax[j].set_title(title, fontsize=9); ax[j].set_xlabel("Steps")
ax[0].legend(fontsize=7)
fig.suptitle("SAC N=1 training diagnostics (5 seeds) - Real Data", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sac_n1_diagnostics_real.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(5, 4, figsize=(14, 14))
for i, s in enumerate(seeds):
    for j, (k, title) in enumerate(cols):
        if k in data[s]:
            x, y = data[s][k]
            ax[i, j].plot(x, y, lw=0.6, color=colors[s])
        if i == 0: ax[i, j].set_title(title, fontsize=9)
        if j == 0: ax[i, j].set_ylabel(f"seed {s}")
fig.suptitle("SAC N=1 training diagnostics, one row per seed - Real Data", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sac_n1_per_seed_real.png"), dpi=200); plt.close(fig)

print("\nDATASET")
df = pd.read_parquet(r"cache\caiso_2024_2026_5min.parquet")
print("rows:", len(df), "columns:", list(df.columns)[:8], "index type:", type(df.index).__name__)
ts = None
for c in df.columns:
    if "time" in c.lower() or "date" in c.lower() or "interval" in c.lower():
        ts = pd.to_datetime(df[c]); break
if ts is None and isinstance(df.index, pd.DatetimeIndex): ts = pd.Series(df.index)
if ts is not None:
    ts = ts.reset_index(drop=True)
    n = len(ts); a = 192095; b = a + 8928
    print("first:", ts.iloc[0], " last:", ts.iloc[-1], " distinct days:", ts.dt.date.nunique())
    print("train :", ts.iloc[0], "->", ts.iloc[a-1])
    print("valid :", ts.iloc[a], "->", ts.iloc[b-1])
    print("test  :", ts.iloc[b], "->", ts.iloc[-1], f"({n-b} steps)")
    gaps = ts.diff().dropna(); print("largest gap:", gaps.max(), " gaps > 5 min:", int((gaps > pd.Timedelta(minutes=5)).sum()))
else:
    print("No time column found; send me the columns line above.")
