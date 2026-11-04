"""
Turkiye dis test seti degerlendirmesi (sifir atis / zero-shot).
Egitilmis modeller HIC degistirilmeden Turkiye izlerine uygulanir.

Raporlananlar (her deney ve her model icin):
  - Ana sonuc: ortak kalite filtresi (snr_db >= MIN_SNR_DB, kirpilmamis), iz ve olay duzeyi,
    olay bazli bootstrap %95 guven araliklari
  - Buyukluk araliklari, donemler (2023 artcilari / 2024-2026), etiket duyarliligi (katalog ML)
  - Electronics (Celik, 2026) ile karsilastirilabilir SNR bantlari (ayni SNR tanimi, filtresiz)
  - KOERI istasyon ML'leri ile karsilastirma (analistin kendi istasyon dagilimi = alt sinir)
  - Yonlendirme (pattern submodel): S varisi olan izlere tam model, olmayanlara S-P'siz model
    (--exclude-features sp_time_s ile egitilen ikiz deney varsa otomatik; adinda "_ROUTED")
  - Az ornekli sapma duzeltmesi: kronolojik ilk n depremden sabit kayma tahmini; degerlendirme
    yalnizca ilk 100 depremden SONRAKI depremlerde (tum n'ler icin ayni kume)

Kullanim:
    Spyder: F5 (results/ altindaki tum deneyler)
    Tek deney: runfile(..., args='--runs train-STEAD+INSTANCE+CEED_cap40000_event_diverse_nodist_seed42')
Cikti: DataBase/results/turkey_eval/
"""

# Windows: farkli paketlerin getirdigi iki OpenMP kutuphanesi ayni surece yuklenince Python
# cokebiliyor ("OMP: Error #15"). numpy'den ONCE ayarlanmali.
import os  # noqa: E402
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from common import cfg
from train_models import MAG_BINS, bootstrap_ci, metrics
from tune_ensembles import PHYS_FEATS

SNR_BANDS = [-np.inf, 5, 10, 20, np.inf]          # Celik (2026) bantlari (snr_ref_db)
FEWSHOT_N = [10, 25, 50, 100]


# ------------------------------------------------------------------ tahminler
def predict_all(bundle, df, best_knora=None):
    feats = bundle["features"]
    X = df[feats].values
    models, names = bundle["models"], bundle["stack_names"]
    spec = bundle.get("spec", {})
    preds = {}
    for ref in ("PD_SCALING", "RIDGE"):
        if ref in models:
            m, cols = models[ref]
            preds[ref] = m.predict(df[cols].values)
    P = []
    for n in names:
        cols = spec.get(n, {}).get("cols", list(range(len(feats))))
        p = models[n].predict(X[:, cols])
        preds[n] = p
        P.append(p)
    P = np.column_stack(P)
    preds["AVG"] = P.mean(1)
    if "STACK_META" in models:
        preds["STACK"] = models["STACK_META"].predict(P)
    if "BLEND_META" in models:
        preds["BLEND"] = models["BLEND_META"].predict(P)
    if "KNORA" in models:
        kn = models["KNORA"]
        Zq = kn["space"].transform(X)
        err = np.abs(kn["oof_dsel"] - kn["y_dsel"][:, None])
        configs = {"KNORA_E": ("E", "all", kn["k"], kn["tol"]), "KNORA_U": ("U", "all", kn["k"], 1)}
        if best_knora:
            kind, space, k, par = best_knora
            configs["KNORA_TUNED"] = (kind, space, k, par)
        for label, (kind, space, k, par) in configs.items():
            idx = ([feats.index(f) for f in PHYS_FEATS if f in feats] if space == "phys"
                   else list(range(len(feats))))
            preds[label] = knora(kn["Z_dsel"][:, idx], err, Zq[:, idx], P, k, kind, par)
    return preds


def knora(Z_dsel, err, Zq, P, k, kind, par):
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=-1).fit(Z_dsel)
    out = np.empty(len(Zq))
    for i in range(0, len(Zq), 5000):
        _, nb = nn.kneighbors(Zq[i:i + 5000])
        loc = err[nb].mean(axis=1)
        Pb = P[i:i + 5000]
        if kind == "E":
            sel = loc <= loc.min(1, keepdims=True) * (1 + par)
            out[i:i + 5000] = (Pb * sel).sum(1) / sel.sum(1)
        else:
            w = 1.0 / (loc + 1e-6) ** par
            out[i:i + 5000] = (Pb * w).sum(1) / w.sum(1)
    return out


def parse_best_knora(root):
    path = os.path.join(root, "results", "ensemble_tuning_routed", "best_config.json")
    if not os.path.exists(path):                       # yonlendirmeli ayar yoksa eskisi
        path = os.path.join(root, "results", "ensemble_tuning", "best_config.json")
    if not os.path.exists(path):
        return None
    key = json.load(open(path))["best_knora"]          # or. KNORA_U|space=all|k=100|pow=4
    parts = key.split("|")
    kind = parts[0].split("_")[1]
    space = parts[1].split("=")[1]
    k = int(parts[2].split("=")[1])
    par = float(parts[3].split("=")[1])
    return kind, space, k, par


# ------------------------------------------------------------------ metrikler
def row(name, sub, y, p, ev, extra):
    r = metrics(y, p)
    (lo, hi), _ = bootstrap_ci(y, p, ev, seed=cfg("SEED", 42))
    r.update(model=name, subset=sub, MAE_lo=lo, MAE_hi=hi, **extra)
    return r


def event_level(df, p, label):
    e = pd.DataFrame({"e": df["event_id"].values, "y": df[label].values, "p": p,
                      "t": df["origin_time"].values})
    return e.groupby("e").agg(y=("y", "first"), p=("p", "median"), t=("t", "first")).reset_index()


NOSP_SUFFIX = "_excl-sp_time_s"


def route(full, nosp, has_s):
    """Pattern submodel: S varisi olan izlere tam model, olmayanlara S-P'siz model."""
    return {m: np.where(has_s, full[m], nosp[m]) for m in full if m in nosp}


def evaluate_run(run, preds_qc, preds_all, df_qc, df_all):
    rows = []
    y = df_qc["ML_bulletin"].values
    ev_id = df_qc["event_id"].values
    bins = pd.cut(df_qc["ML_bulletin"], MAG_BINS, include_lowest=True)
    period = np.where(df_qc["origin_time"].str[:4] == "2023", "2023", "2024-2026")
    has_sta = df_qc["sta_ml"].notna().values

    for name, p in preds_qc.items():
        rows.append(row(name, "all", y, p, ev_id, {"level": "trace"}))
        evt = event_level(df_qc, p, "ML_bulletin")
        rows.append(row(name, "all", evt["y"].values, evt["p"].values, evt["e"].values, {"level": "event"}))
        if df_qc["catalog_ML"].notna().any():
            yc = df_qc["catalog_ML"].values
            ok = ~np.isnan(yc)
            rows.append(row(name, "label=catalog_ML", yc[ok], p[ok], ev_id[ok], {"level": "trace"}))
        for b in bins.cat.categories:
            m = (bins == b).values
            if m.sum() >= 20:
                rows.append(row(name, f"ML {b}", y[m], p[m], ev_id[m], {"level": "trace"}))
        for per in ("2023", "2024-2026"):
            m = period == per
            if m.sum() >= 20:
                rows.append(row(name, f"period {per}", y[m], p[m], ev_id[m], {"level": "trace"}))
        if has_sta.sum() >= 20:
            rows.append(row(name, "vs station ML", df_qc["sta_ml"].values[has_sta], p[has_sta],
                            ev_id[has_sta], {"level": "trace"}))

    # Celik (2026) ile karsilastirma: ayni SNR tanimi (snr_ref_db), kalite filtresi yok
    bands = pd.cut(df_all["snr_ref_db"], SNR_BANDS)
    for name, p in preds_all.items():
        for b in bands.cat.categories:
            m = (bands == b).values
            if m.sum() >= 20:
                rows.append(row(name, f"SNRref {b}", df_all["ML_bulletin"].values[m], p[m],
                                df_all["event_id"].values[m], {"level": "trace"}))

    # Az ornekli sapma duzeltmesi (olay duzeyi)
    few = []
    for name, p in preds_qc.items():
        evt = event_level(df_qc, p, "ML_bulletin").sort_values("t").reset_index(drop=True)
        test = evt.iloc[max(FEWSHOT_N):]
        few.append({"model": name, "n_calib": 0, "offset": 0.0,
                    "MAE": float(np.mean(np.abs(test["p"] - test["y"]))),
                    "bias": float(np.mean(test["p"] - test["y"])), "n_test_events": len(test)})
        for n in FEWSHOT_N:
            off = float(np.median(evt["p"].iloc[:n] - evt["y"].iloc[:n]))
            q = test["p"] - off
            few.append({"model": name, "n_calib": n, "offset": off,
                        "MAE": float(np.mean(np.abs(q - test["y"]))),
                        "bias": float(np.mean(q - test["y"])), "n_test_events": len(test)})

    res = pd.DataFrame(rows)
    res.insert(0, "run", run)
    fewdf = pd.DataFrame(few)
    fewdf.insert(0, "run", run)
    pred_df = df_qc[["event_id", "station", "origin_time", "ML_bulletin", "distance_km", "snr_db"]].copy()
    for n, p in preds_qc.items():
        pred_df[f"pred_{n}"] = p
    return res, fewdf, pred_df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=cfg("ROOT"), required=cfg("ROOT") is None)
    ap.add_argument("--koeri", default=cfg("KOERI"), required=cfg("KOERI") is None)
    ap.add_argument("--runs", nargs="*", help="Bos = results/ altindaki tum deneyler")
    ap.add_argument("--min-snr", type=float, default=cfg("MIN_SNR_DB", 20.0))
    args = ap.parse_args(argv)

    tr = pd.read_csv(os.path.join(args.koeri, "turkey_features.csv.gz"), dtype={"event_id": str})
    tr = tr.replace([np.inf, -np.inf], np.nan)
    tr["log_hypo_dist"] = np.log10(np.sqrt(tr["distance_km"] ** 2 + tr["depth_km"].clip(lower=0) ** 2) + 1)
    tr["origin_time"] = tr["origin_time"].astype(str)
    df_all = tr[~tr["clipped"].astype(bool)].reset_index(drop=True)
    df_qc = df_all[df_all["snr_db"] >= args.min_snr].reset_index(drop=True)
    print(f"Turkiye: {len(tr):,} iz | kirpilmamis: {len(df_all):,} | "
          f"kalite filtresi (snr_db >= {args.min_snr}): {len(df_qc):,} iz, {df_qc['event_id'].nunique():,} deprem")

    runs = args.runs or sorted(os.path.basename(os.path.dirname(p)) for p in
                               glob.glob(os.path.join(args.root, "results", "train-*", "models.joblib")))
    best_knora = parse_best_knora(args.root)
    print(f"Degerlendirilecek deney: {len(runs)} | ayarlanmis KNORA: {best_knora}")

    out_dir = os.path.join(args.root, "results", "turkey_eval")
    os.makedirs(out_dir, exist_ok=True)
    all_res, all_few = [], []
    cache = {}

    def run_and_store(label, pq, pa):
        res, few, pred = evaluate_run(label, pq, pa, df_qc, df_all)
        pred.to_csv(os.path.join(out_dir, f"pred_{label}.csv.gz"), index=False)
        all_res.append(res)
        all_few.append(few)
        main_rows = res[(res["subset"] == "all")]
        print(main_rows.pivot_table(index="model", columns="level", values="MAE").round(3)
              .sort_values("trace").to_string())

    for run in runs:
        print(f"\n--- {run}", flush=True)
        bundle = joblib.load(os.path.join(args.root, "results", run, "models.joblib"))
        pq, pa = predict_all(bundle, df_qc, best_knora), predict_all(bundle, df_all, best_knora)
        cache[run] = (pq, pa)
        run_and_store(run, pq, pa)

    # Yonlendirme: tam model (S varsa) + S-P'siz alt model (S yoksa)
    for run in runs:
        if NOSP_SUFFIX in run:
            continue
        twin = run.replace("_seed", f"{NOSP_SUFFIX}_seed")
        if twin not in cache:
            continue
        label = run.replace("_seed", "_ROUTED_seed")
        print(f"\n--- {label} (S varsa tam model, yoksa S-P'siz model)", flush=True)
        (fq, fa), (nq, na) = cache[run], cache[twin]
        run_and_store(label, route(fq, nq, df_qc["has_S"].values.astype(bool)),
                      route(fa, na, df_all["has_S"].values.astype(bool)))

    res = pd.concat(all_res, ignore_index=True)
    few = pd.concat(all_few, ignore_index=True)
    res.to_csv(os.path.join(out_dir, "metrics_turkey.csv"), index=False)
    few.to_csv(os.path.join(out_dir, "fewshot_turkey.csv"), index=False)

    print("\n=== OZET: iz duzeyi MAE [%95 GA], tum deneyler ===")
    s = res[(res["subset"] == "all") & (res["level"] == "trace")].copy()
    s["MAE_GA"] = s.apply(lambda r: f"{r.MAE:.3f} [{r.MAE_lo:.3f}-{r.MAE_hi:.3f}]", axis=1)
    print(s.pivot(index="model", columns="run", values="MAE_GA").to_string())
    print("\n=== Az ornekli sapma duzeltmesi (olay duzeyi MAE) ===")
    print(few.pivot_table(index=["run", "model"], columns="n_calib", values="MAE").round(3).to_string())
    print(f"\nKaydedildi: {out_dir}")


if __name__ == "__main__":
    main()