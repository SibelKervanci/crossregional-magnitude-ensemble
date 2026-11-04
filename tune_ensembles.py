"""
Ensemble ve KNORA parametrelerinin BOLGELER ARASI dogrulama ile ayarlanmasi.
(leave-one-domain-out validation; Gulrajani & Lopez-Paz, 2021)

Fikir: Parametreler, bir veri setiyle egitilen modellerin DIGER veri setinin
DOGRULAMA bolmesindeki performansina gore secilir. Boylece "bolge icinde en iyi"
degil, "bilinmeyen bolgeye en iyi tasinan" ayar secilir. Test setlerine ve
Turkiye verisine dokunulmaz. Temel modeller yeniden egitilmez (kayitli modeller).

Kullanim:
    python tune_ensembles.py --root ~/work/deprem/DataBase \
        --runs train-STEAD_cap40000_event_diverse_nodist_seed42 \
               train-INSTANCE_cap40000_event_diverse_nodist_seed42
Yonlendirme (varsayilan): her LODO deneyinin S-P'siz ikizi varsa, S varisi olmayan izlere
ikiz modelin tahminleri kullanilir (pattern submodel). Kapatmak icin --no-route.

Cikti:
    DataBase/results/ensemble_tuning_routed/  (yonlendirmeli)
    DataBase/results/ensemble_tuning/         (--no-route)
"""

# Windows: farkli paketlerin getirdigi iki OpenMP kutuphanesi ayni surece yuklenince Python
# cokebiliyor ("OMP: Error #15"). numpy'den ONCE ayarlanmali.
import os  # noqa: E402
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import itertools
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from common import cfg


K_GRID = [10, 25, 50, 100, 200]
TOL_GRID = [0.0, 0.05, 0.10, 0.20, 0.50]
POW_GRID = [1, 2, 4]
PHYS_FEATS = ["sp_time_s", "log_PGV", "log_PGD", "log_PGA", "dur_5_95_s",
              "dom_freq", "spec_centroid", "Pd_3s", "tauc_3s"]


def base_predictions(bundle, X):
    names = bundle["stack_names"]
    cols = [bundle["spec"][n]["cols"] for n in names]
    return np.column_stack([bundle["models"][n].predict(X[:, c]) for n, c in zip(names, cols)])


def knora_all(Z_dsel, err, Zq, P, k_max):
    """En buyuk k icin komsulari bir kez bul; kucuk k'lar ilk k komsudan elde edilir."""
    nn = NearestNeighbors(n_neighbors=k_max, algorithm="brute", n_jobs=-1).fit(Z_dsel)
    out = {}
    for i in range(0, len(Zq), 5000):
        _, nb = nn.kneighbors(Zq[i:i + 5000])
        Pb = P[i:i + 5000]
        for k in K_GRID:
            loc = err[nb[:, :k]].mean(axis=1)
            for tol in TOL_GRID:
                sel = loc <= loc.min(1, keepdims=True) * (1 + tol)
                out.setdefault(("E", k, tol), []).append((Pb * sel).sum(1) / sel.sum(1))
            for pw in POW_GRID:
                w = 1.0 / (loc + 1e-6) ** pw
                out.setdefault(("U", k, pw), []).append((Pb * w).sum(1) / w.sum(1))
    return {key: np.concatenate(v) for key, v in out.items()}


def default_runs():
    """config.py'deki veri setleri icin tek veri setiyle egitilmis (LODO) ana deney klasorleri."""
    return [f"train-{d}_cap40000_event_diverse_nodist_seed{cfg('SEED', 42)}" for d in cfg("DATASETS", [])]


NOSP_SUFFIX = "_excl-sp_time_s"


def method_predictions(bundle, data):
    """Bir deneyin tum yontem/ayar tahminleri (temel modeller, AVG, STACK, BLEND, KNORA izgarasi)."""
    feats = bundle["features"]
    X = data[feats].values
    P = base_predictions(bundle, X)
    res = {"AVG": P.mean(1)}
    if "STACK_META" in bundle["models"]:
        res["STACK"] = bundle["models"]["STACK_META"].predict(P)
    if "BLEND_META" in bundle["models"]:
        res["BLEND"] = bundle["models"]["BLEND_META"].predict(P)
    for j, n in enumerate(bundle["stack_names"]):
        res[n] = P[:, j]
    kn = bundle["models"]["KNORA"]
    Zq = kn["space"].transform(X)
    err = np.abs(kn["oof_dsel"] - kn["y_dsel"][:, None])
    phys_idx = [feats.index(f) for f in PHYS_FEATS if f in feats]
    for space_name, idx in (("all", list(range(len(feats)))), ("phys", phys_idx)):
        out = knora_all(kn["Z_dsel"][:, idx], err, Zq[:, idx], P, max(K_GRID))
        for (kind, k, par), p in out.items():
            key = f"KNORA_{kind}|space={space_name}|k={k}|" + ("tol" if kind == "E" else "pow") + f"={par}"
            res[key] = p
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=cfg("ROOT"), required=cfg("ROOT") is None)
    ap.add_argument("--runs", nargs="+", default=default_runs(), help="LODO deney klasorleri")
    ap.add_argument("--min-snr", type=float, default=cfg("MIN_SNR_DB", 20.0))
    ap.add_argument("--no-route", dest="route", action="store_false",
                    help="Yonlendirmeyi kapat (yalnizca tam modeller)")
    args = ap.parse_args(argv)
    root = os.path.expanduser(args.root)

    df = pd.read_csv(os.path.join(root, "features", "features_all.csv.gz"), low_memory=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["snr_db"] >= args.min_snr]      # train_models.py ile ayni kalite filtresi
    ex_path = os.path.join(root, "harmonized", "exclude_events.csv")
    if os.path.exists(ex_path):
        df = df[~df["event_id"].isin(set(pd.read_csv(ex_path)["event_id"]))]

    rows = []
    for run in args.runs:
        bundle = joblib.load(os.path.join(root, "results", run, "models.joblib"))
        src = bundle["args"]["train_datasets"]
        tgt_dev = df[(~df["dataset"].isin(src)) & (df["split"] == "dev")]
        if tgt_dev.empty:
            print(f"{run}: hedef dogrulama verisi yok, atlandi")
            continue
        y = tgt_dev["magnitude"].values
        tgt = "+".join(sorted(tgt_dev["dataset"].unique()))
        res = method_predictions(bundle, tgt_dev)
        twin = run.replace("_seed", f"{NOSP_SUFFIX}_seed")
        twin_path = os.path.join(root, "results", twin, "models.joblib")
        mode = "tam model"
        if args.route and os.path.exists(twin_path):
            # Yonlendirme: S varisi olan izlere tam model, olmayanlara S-P'siz ikiz
            res_nosp = method_predictions(joblib.load(twin_path), tgt_dev)
            has_s = tgt_dev["sp_time_s"].notna().values
            res = {k: np.where(has_s, res[k], res_nosp[k]) for k in res if k in res_nosp}
            mode = f"yonlendirilmis (S yok: {(~has_s).mean():.1%})"
        print(f"\n{run}\n  kaynak={src} -> hedef dogrulama={tgt} ({len(y):,} iz), {mode}", flush=True)
        for key, p in res.items():
            rows.append({"run": run, "target": tgt, "method": key,
                         "MAE": float(np.mean(np.abs(p - y))), "bias": float(np.mean(p - y))})

    grid = pd.DataFrame(rows)
    out_dir = os.path.join(root, "results", "ensemble_tuning" + ("_routed" if args.route else ""))
    os.makedirs(out_dir, exist_ok=True)
    grid.to_csv(os.path.join(out_dir, "grid.csv"), index=False)

    # Secim olcutu: tum hedef bolgelerdeki ortalama MAE (bolgeler arasi genelleme)
    summary = grid.groupby("method")["MAE"].agg(["mean", "max"]).sort_values("mean")
    summary.columns = ["MAE_ortalama", "MAE_en_kotu"]
    summary.to_csv(os.path.join(out_dir, "summary.csv"))
    print("\n=== Bolgeler arasi dogrulama: en iyi 15 yontem/ayar ===")
    print(summary.head(15).round(4).to_string())
    print("\n=== Referans yontemler ===")
    ref = [m for m in summary.index if "|" not in m]
    print(summary.loc[ref].round(4).to_string())

    best_knora = summary[summary.index.str.startswith("KNORA")].index[0]
    best = {"best_overall": summary.index[0], "best_knora": best_knora,
            "best_knora_mae": float(summary.loc[best_knora, "MAE_ortalama"]),
            "criterion": "mean cross-domain dev MAE", "runs": args.runs}
    with open(os.path.join(out_dir, "best_config.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"\nEn iyi KNORA ayari: {best_knora}")
    print(f"Kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()