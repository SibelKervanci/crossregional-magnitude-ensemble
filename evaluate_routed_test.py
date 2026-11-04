"""
Bolge ici ve bolgeler arasi test setlerinde yonlendirme (pattern submodel) degerlendirmesi.
Her deney icin tam model ile S-P'siz ikizinin test tahminleri birlestirilir:
S varisi olan izlere tam model, olmayanlara S-P'siz model.

Raporlananlar (her deney, test seti ve model icin):
  - tam / S-P'siz / yonlendirilmis tahminlerin MAE'si (iz ve olay duzeyi, %95 GA)
  - S varisi olan ve olmayan izlerde ayri MAE ve sistematik sapma

Kullanim: Spyder'da F5 (S-P'siz deneyler tamamlandiktan sonra)
Cikti: DataBase/results/routed_tests/
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import glob  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import cfg  # noqa: E402
from train_models import bootstrap_ci, metrics  # noqa: E402

NOSP = "_excl-sp_time_s"


def score(rows, run, variant, test, y, p, has_s, seed):
    for ds in sorted(test["dataset"].unique()):
        m = (test["dataset"] == ds).values
        ev = test["event_id"].values[m]
        r = metrics(y[m], p[m])
        (lo, hi), _ = bootstrap_ci(y[m], p[m], ev, seed)
        e = pd.DataFrame({"e": ev, "y": y[m], "p": p[m]}).groupby("e").agg(y=("y", "first"), p=("p", "median"))
        out = {"run": run, "variant": variant, "test_set": ds, "n": int(m.sum()),
               "frac_noS": float((~has_s[m]).mean()), "MAE": r["MAE"], "MAE_lo": lo, "MAE_hi": hi,
               "bias": r["bias"], "MAE_event": float(np.mean(np.abs(e["p"] - e["y"])))}
        for flag, name in ((True, "S"), (False, "noS")):
            k = m & (has_s == flag)
            if k.sum() >= 20:
                out[f"MAE_{name}"] = float(np.mean(np.abs(p[k] - y[k])))
                out[f"bias_{name}"] = float(np.mean(p[k] - y[k]))
        rows.append(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=cfg("ROOT"), required=cfg("ROOT") is None)
    ap.add_argument("--models", nargs="*", default=["STACK", "AVG", "KNORA_E", "XGB", "LGBM", "CAT", "MLP", "RF", "ET"])
    args = ap.parse_args(argv)
    root, seed = args.root, cfg("SEED", 42)

    feats = pd.read_csv(os.path.join(root, "features", "features_all.csv.gz"),
                        usecols=["dataset", "trace_name", "sp_time_s"])
    feats["has_S"] = feats["sp_time_s"].notna()
    feats = feats.drop(columns="sp_time_s")

    out_dir = os.path.join(root, "results", "routed_tests")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for nosp_dir in sorted(glob.glob(os.path.join(root, "results", f"train-*{NOSP}_seed*"))):
        twin = os.path.basename(nosp_dir)
        run = twin.replace(NOSP, "")
        full_path = os.path.join(root, "results", run, "predictions_test.csv.gz")
        if not os.path.exists(full_path):
            continue
        print(f"--- {run}", flush=True)
        full = pd.read_csv(full_path, low_memory=False)
        nosp = pd.read_csv(os.path.join(nosp_dir, "predictions_test.csv.gz"), low_memory=False)
        test = full.merge(nosp[["dataset", "trace_name"] + [c for c in nosp if c.startswith("pred_")]],
                          on=["dataset", "trace_name"], suffixes=("", "_nosp"))
        test = test.merge(feats, on=["dataset", "trace_name"], how="left")
        has_s = test["has_S"].fillna(False).values.astype(bool)
        y = test["magnitude"].values
        for m in args.models:
            if f"pred_{m}" not in test or f"pred_{m}_nosp" not in test:
                continue
            pf, pn = test[f"pred_{m}"].values, test[f"pred_{m}_nosp"].values
            score(rows, run, f"{m} tam", test, y, pf, has_s, seed)
            score(rows, run, f"{m} S-P'siz", test, y, pn, has_s, seed)
            score(rows, run, f"{m} yonlendirilmis", test, y, np.where(has_s, pf, pn), has_s, seed)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "metrics_routed_tests.csv"), index=False)
    pd.set_option("display.width", 250)
    for run, g in res.groupby("run"):
        print(f"\n=== {run} ===")
        cols = ["variant", "test_set", "frac_noS", "MAE", "MAE_lo", "MAE_hi", "MAE_S", "MAE_noS", "bias_noS"]
        print(g[[c for c in cols if c in g]].round(3).to_string(index=False))
    print(f"\nKaydedildi: {out_dir}")


if __name__ == "__main__":
    main()