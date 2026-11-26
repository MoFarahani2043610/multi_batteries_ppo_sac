import csv
rows = list(csv.DictReader(open("results_log.csv", newline="", encoding="utf-8")))
cols = list(rows[0].keys())
fix = {"0": "27.14", "1": "27.28", "2": "29.51", "3": "27.89", "4": "28.36"}
def add(r, txt):
    r["notes"] = (r["notes"] + " | " if r["notes"] else "") + txt
n = dict(seed=0, path=0, prelim=0, sweep=0, back=0)
for r in rows:
    if r["algo"] == "SAC" and r["n_batteries"] == "1" and r["config_path"].endswith("sac_n1.yaml") and r["seed"] in fix:
        add(r, "corrected per-seed mean (was %s); per-seed mean over 10 episodes" % r["eval_mean_return"])
        r["eval_mean_return"] = fix[r["seed"]]; n["seed"] += 1
    if r["config_path"] == "test_traintest_split_seed1.py":
        r["config_path"] = "test_traintest_split.py"; n["path"] += 1
    if int(r["run_id"]) <= 18:
        add(r, "backfilled: timestamp is the time of backfill, not of the run"); n["back"] += 1
    if "prelim" in r["config_path"]:
        add(r, "PRELIMINARY 200K-step run, NOT used in the thesis; SAC rows may have used price_ref 52.0 (config edited afterwards)"); n["prelim"] += 1
    if r["notes"].startswith("M3 sweep cell"):
        add(r, "sweep-internal evaluation on default batteries (not valid for the thesis); thesis values are in m3_reeval_episodes.csv"); n["sweep"] += 1
with open("results_log.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(n, "total rows:", len(rows))
