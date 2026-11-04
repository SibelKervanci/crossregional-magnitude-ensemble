"""
Hipotez testi: LightGBM/CatBoost'un bolge disi cokusu S-P suresinin eksik olmasindan mi?
  1) Egitim veri setlerinde S varisi olan izlerin orani
  2) Turkiye'de, S varisi olan ve olmayan izlerde ayri ayri MAE ve sistematik sapma
Kullanim: Spyder'da F5
"""
import os

import pandas as pd

from common import cfg

ROOT, KOERI = cfg("ROOT"), cfg("KOERI")
TAG = "_cap40000_event_diverse_nodist_seed42"
RUNS = ["CEED", "STEAD", "INSTANCE", "STEAD+INSTANCE+CEED"]
MODELS = ["XGB", "LGBM", "CAT", "RF", "MLP", "STACK"]

f = pd.read_csv(os.path.join(ROOT, "features", "features_all.csv.gz"), usecols=["dataset", "sp_time_s"])
print("1) Egitim verisinde S varisi olan izlerin orani:")
print(f.groupby("dataset")["sp_time_s"].apply(lambda s: s.notna().mean()).round(3).to_string())

tf = pd.read_csv(os.path.join(KOERI, "turkey_features.csv.gz"), dtype={"event_id": str},
                 usecols=["event_id", "station", "has_S"])
rows = []
for run in RUNS:
    p = pd.read_csv(os.path.join(ROOT, "results", "turkey_eval", f"pred_train-{run}{TAG}.csv.gz"),
                    dtype={"event_id": str}).merge(tf, on=["event_id", "station"], how="left")
    for m in MODELS:
        for has_s, g in p.groupby("has_S"):
            e = g[f"pred_{m}"] - g["ML_bulletin"]
            rows.append({"egitim": run, "model": m, "S_var": bool(has_s), "n": len(g),
                         "MAE": e.abs().mean(), "sapma": e.mean()})
res = pd.DataFrame(rows)
print("\n2) Turkiye: S varisi olan (True) ve olmayan (False) izlerde MAE ve sapma")
print(res.pivot_table(index=["egitim", "model"], columns="S_var", values=["MAE", "sapma"]).round(3).to_string())