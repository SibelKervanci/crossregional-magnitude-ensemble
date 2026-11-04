"""
Model egitimi ve degerlendirme (sizintisiz, hakem itirazlarina karsi tasarlanmis).

Modeller: Pd olcekleme (klasik), Ridge, RF, XGBoost, LightGBM, CatBoost.
Havuz (--pool diverse): RF, ET, KNN, MLP, XGB, LGBM, CAT + oznitelik uzmanlari LGBM_P (erken P) ve LGBM_F (tam pencere).
Ensemble: AVG (ortalama), BLEND, STACK (OOF), KNORA-E ve KNORA-U (dinamik secim, regresyon).
Tasarim ilkeleri:
  - Test setleri hicbir karar icin kullanilmaz (erken durdurma, ayarlama, stacking,
    alt orneklem siniri yalnizca egitim/dogrulama ile belirlenir).
  - Bolme olay bazlidir (--split-mode event) veya zamana goredir (--split-mode time).
  - audit_leakage.py'nin urettigi exclude_events.csv otomatik uygulanir.
  - Stacking ust modeli, olay bazli GroupKFold ile uretilen kat-disi tahminlerle egitilir.
  - Sonuclar olay bazli bootstrap ile %95 guven araliklariyla raporlanir.

Ornekler:
    python train_models.py --root ~/work/deprem/DataBase                       # ana deney
    python train_models.py --root ~/work/deprem/DataBase --cap 0               # dogal dagilim
    python train_models.py --root ~/work/deprem/DataBase --with-distance       # uzaklik ablasyonu
    python train_models.py --root ~/work/deprem/DataBase --train-datasets STEAD   # LODO
    python train_models.py --root ~/work/deprem/DataBase --split-mode time     # zamana gore bolme
    python train_models.py --root ~/work/deprem/DataBase --tune 40             # Optuna ayari
    python train_models.py --root ~/work/deprem/DataBase --exclude-features sp_time_s   # S-P'siz alt model
"""

# Windows: farkli paketlerin getirdigi iki OpenMP kutuphanesi ayni surece yuklenince Python
# cokebiliyor ("OMP: Error #15"). numpy'den ONCE ayarlanmali.
import os  # noqa: E402
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import os
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from common import cfg


warnings.filterwarnings("ignore", message=".*eval_set.*")

MAG_BINS = [0, 1, 2, 3, 4, 5, 10]
P_FEATS = [f"{k}_{w}s" for w in (1, 2, 3) for k in ("Pd", "Pv", "Pa", "PvVec", "IV2", "tauc")]
FULL_FEATS = ["log_PGV", "log_PGD", "log_PGA", "log_PGV_h", "log_CAV", "log_arias",
              "log_energy_v", "dur_5_95_s"]
SPEC_FEATS = ["dom_freq", "spec_centroid", "spec_entropy", "log_ratio_low_mid", "log_ratio_mid_high"]
BASE_FEATURES = P_FEATS + FULL_FEATS + SPEC_FEATS + ["sp_time_s"]
# Bilincli olarak DISARIDA birakilanlar: snr_db, snr_ref_db, log_noise_rms
# (istasyon/ag gurultusunu yansitir; genellemeyi bozar).
GBM_NAMES = ("XGB", "LGBM", "CAT")
N_BOOT = 1000


# ---------------------------------------------------------------- veri hazirligi
def time_based_split(df, trace_index_path):
    """Her veri setinde depremleri olus zamanina gore 70/15/15 boler (eski -> yeni)."""
    ti = pd.read_csv(trace_index_path, usecols=["dataset", "sb_index", "start_time"], low_memory=False)
    df = df.drop(columns="split").merge(ti, on=["dataset", "sb_index"], how="left")
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, format="mixed", errors="coerce")
    ev_time = df.groupby("event_id")["start_time"].min()
    split = {}
    for ds in df["dataset"].unique():
        t = ev_time[df.loc[df["dataset"] == ds, "event_id"].unique()].sort_values()
        n = len(t)
        lab = np.array(["train"] * n, dtype=object)
        lab[int(0.70 * n):int(0.85 * n)] = "dev"
        lab[int(0.85 * n):] = "test"
        split.update(dict(zip(t.index, lab)))
    df["split"] = df["event_id"].map(split)
    return df.drop(columns="start_time")


def balanced_sample(df, cap, seed):
    if cap <= 0:
        return df
    mbin = pd.cut(df["magnitude"], MAG_BINS, include_lowest=True)
    return pd.concat([g.sample(min(len(g), cap), random_state=seed)
                      for _, g in df.groupby([df["dataset"], mbin], observed=True)])


# ---------------------------------------------------------------- modeller
def default_params(seed, n_jobs):
    return {
        "XGB": dict(n_estimators=10000, learning_rate=0.05, max_depth=8, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                    early_stopping_rounds=100, n_jobs=n_jobs, random_state=seed),
        "LGBM": dict(n_estimators=10000, learning_rate=0.05, num_leaves=127, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                     n_jobs=n_jobs, random_state=seed, verbose=-1),
        "CAT": dict(iterations=10000, learning_rate=0.08, depth=8, l2_leaf_reg=3,
                    random_seed=seed, thread_count=n_jobs, verbose=0, od_type="Iter", od_wait=100),
    }


def build_gbm(name, params):
    return {"XGB": XGBRegressor, "LGBM": LGBMRegressor, "CAT": CatBoostRegressor}[name](**params)


def fit_gbm(name, model, Xtr, ytr, Xdv, ydv):
    if name == "XGB":
        model.fit(Xtr, ytr, eval_set=[(Xdv, ydv)], verbose=False)
        return model, model.best_iteration + 1
    if name == "LGBM":
        model.fit(Xtr, ytr, eval_set=[(Xdv, ydv)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return model, model.best_iteration_
    model.fit(Xtr, ytr, eval_set=(Xdv, ydv))
    return model, model.get_best_iteration() + 1


def fixed_size(name, params, best):
    """Erken durdurmada bulunan agac sayisiyla sabit model (stacking katlari icin)."""
    p = dict(params)
    if name == "XGB":
        p.update(n_estimators=best, early_stopping_rounds=None)
    elif name == "LGBM":
        p.update(n_estimators=best)
    else:
        p.pop("od_type", None)
        p.pop("od_wait", None)
        p.update(iterations=best)
    return build_gbm(name, p)


def tune_gbm(name, base, Xtr, ytr, Xdv, ydv, n_trials, seed):
    """Optuna: yalnizca dogrulama seti MAE'si ile; test setine dokunmaz."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = dict(base)
        if name == "XGB":
            p.update(max_depth=trial.suggest_int("max_depth", 4, 12),
                     learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                     min_child_weight=trial.suggest_int("min_child_weight", 1, 50, log=True),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0))
        elif name == "LGBM":
            p.update(num_leaves=trial.suggest_int("num_leaves", 15, 511, log=True),
                     learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                     min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0))
        else:
            p.update(depth=trial.suggest_int("depth", 4, 10),
                     learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                     l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True))
        m, _ = fit_gbm(name, build_gbm(name, p), Xtr, ytr, Xdv, ydv)
        return mean_absolute_error(ydv, m.predict(Xdv))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    p = dict(base)
    p.update(study.best_params)
    return p, study.best_value


# ---------------------------------------------------------------- dinamik secim
def knora_regression(Z_dsel, oof, y_dsel, Z_q, P_q, k, tol, n_jobs, batch=5000):
    """KNORA-E ve KNORA-U'nun regresyon uyarlamasi.
    Yerel yetkinlik, DSEL (egitim seti) komsularindaki KAT-DISI tahmin hatasidir;
    boylece modellerin egitimde gordugu orneklerdeki iyimser hatalari kullanilmaz.
      KNORA-E: yerel MAE'si en iyinin (1+tol) katina kadar olan modellerin ortalamasi
      KNORA-U: tum modeller, yerel MAE ile ters orantili agirliklarla
    """
    err = np.abs(oof - y_dsel[:, None])
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=n_jobs).fit(Z_dsel)
    pe, pu = np.empty(len(Z_q)), np.empty(len(Z_q))
    for i in range(0, len(Z_q), batch):
        _, nb = nn.kneighbors(Z_q[i:i + batch])
        loc = err[nb].mean(axis=1)
        P = P_q[i:i + batch]
        sel = loc <= loc.min(axis=1, keepdims=True) * (1 + tol)
        pe[i:i + batch] = (P * sel).sum(1) / sel.sum(1)
        w = 1.0 / (loc + 1e-6)
        pu[i:i + batch] = (P * w).sum(1) / w.sum(1)
    return pe, pu


# ---------------------------------------------------------------- degerlendirme
def metrics(y, p):
    return {"n": len(y), "MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p) if len(y) > 1 else np.nan, "bias": float(np.mean(p - y))}


def bootstrap_ci(y, p, groups, seed, n_boot=N_BOOT):
    """Olay bazli bootstrap: depremler yeniden orneklenir (izler degil)."""
    g = pd.DataFrame({"g": groups, "ae": np.abs(p - y), "se": (p - y) ** 2})
    agg = g.groupby("g").agg(ae=("ae", "sum"), se=("se", "sum"), n=("ae", "size"))
    ae, se, n = agg["ae"].values, agg["se"].values, agg["n"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    mae = ae[idx].sum(1) / n[idx].sum(1)
    rmse = np.sqrt(se[idx].sum(1) / n[idx].sum(1))
    return np.percentile(mae, [2.5, 97.5]), np.percentile(rmse, [2.5, 97.5])


def evaluate(test, preds, train_keys, train_datasets, seed):
    rows, bin_rows = [], []
    y = test["magnitude"].values
    bins = pd.cut(test["magnitude"], MAG_BINS, include_lowest=True)
    sta_key = (test["network"].astype(str) + "." + test["station"].astype(str)).values
    unseen = ~np.isin(sta_key, list(train_keys))
    for name, p in preds.items():
        for ds in sorted(test["dataset"].unique()):
            base = (test["dataset"] == ds).values
            for subset, mask in (("all", base), ("unseen_station", base & unseen)):
                if mask.sum() < 20:
                    continue
                r = metrics(y[mask], p[mask])
                (lo, hi), (rlo, rhi) = bootstrap_ci(y[mask], p[mask], test["event_id"].values[mask], seed)
                r.update(model=name, test_set=ds, subset=subset, level="trace",
                         in_training=ds in train_datasets, MAE_lo=lo, MAE_hi=hi, RMSE_lo=rlo, RMSE_hi=rhi)
                rows.append(r)
            # Olay duzeyi: istasyon tahminlerinin medyani
            ev = pd.DataFrame({"e": test["event_id"].values[base], "y": y[base], "p": p[base]})
            ev = ev.groupby("e").agg(y=("y", "first"), p=("p", "median"))
            r = metrics(ev["y"].values, ev["p"].values)
            (lo, hi), (rlo, rhi) = bootstrap_ci(ev["y"].values, ev["p"].values, ev.index.values, seed)
            r.update(model=name, test_set=ds, subset="all", level="event",
                     in_training=ds in train_datasets, MAE_lo=lo, MAE_hi=hi, RMSE_lo=rlo, RMSE_hi=rhi)
            rows.append(r)
            for b in bins.cat.categories:
                bm = base & (bins == b).values
                if bm.sum() >= 20:
                    rb = metrics(y[bm], p[bm])
                    rb.update(model=name, test_set=ds, mag_bin=str(b))
                    bin_rows.append(rb)
    cols = ["model", "test_set", "level", "subset", "in_training", "n", "MAE", "MAE_lo", "MAE_hi",
            "RMSE", "RMSE_lo", "RMSE_hi", "R2", "bias"]
    return pd.DataFrame(rows)[cols], pd.DataFrame(bin_rows)


# ---------------------------------------------------------------- ana akis
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=cfg("ROOT"), required=cfg("ROOT") is None)
    ap.add_argument("--train-datasets", nargs="+", default=cfg("DATASETS", ["STEAD", "INSTANCE"]))
    ap.add_argument("--cap", type=int, default=40000, help="0 = dogal dagilim")
    ap.add_argument("--split-mode", choices=["event", "time"], default="event")
    ap.add_argument("--with-distance", action="store_true")
    ap.add_argument("--exclude-features", nargs="*", default=[],
                    help="Disarida birakilacak oznitelikler (or. sp_time_s: S-P'siz alt model)")
    ap.add_argument("--min-snr", type=float, default=cfg("MIN_SNR_DB", 20.0),
                    help="Ortak kalite filtresi: dalga formundan hesaplanan snr_db esigi")
    ap.add_argument("--tune", type=int, default=0, help="Optuna deneme sayisi (0 = kapali)")
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--pool", choices=["trees", "diverse"], default="diverse",
                    help="trees: RF+XGB+LGBM+CAT; diverse: + ET, KNN, MLP ve oznitelik uzmanlari")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--des-k", type=int, default=50, help="KNORA komsu sayisi")
    ap.add_argument("--des-tol", type=float, default=0.10, help="KNORA-E tolerans")
    ap.add_argument("--des-max", type=int, default=100000, help="DSEL azami boyutu")
    ap.add_argument("--seed", type=int, default=cfg("SEED", 42))
    ap.add_argument("--n-jobs", type=int, default=cfg("N_JOBS", 16))
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.root)
    df = pd.read_csv(os.path.join(root, "features", "features_all.csv.gz"), low_memory=False)
    df = df.replace([np.inf, -np.inf], np.nan)

    ex_path = os.path.join(root, "harmonized", "exclude_events.csv")
    if os.path.exists(ex_path):
        ex = set(pd.read_csv(ex_path)["event_id"])
        n0 = len(df)
        df = df[~df["event_id"].isin(ex)]
        print(f"Sizinti denetimi: {n0 - len(df):,} iz cikarildi ({len(ex):,} deprem)")
    else:
        print("UYARI: exclude_events.csv yok. Once audit_leakage.py calistirin!")

    n0 = len(df)
    df = df[df["snr_db"] >= args.min_snr]
    print(f"Ortak SNR filtresi (snr_db >= {args.min_snr} dB): {len(df):,}/{n0:,} iz kaldi")
    print(df.groupby("dataset").agg(iz=("trace_name", "size"), olay=("event_id", "nunique")).to_string())

    if args.split_mode == "time":
        df = time_based_split(df, os.path.join(root, "harmonized", "trace_index.csv.gz"))
    assert df.groupby("event_id")["split"].nunique().max() == 1, "Olay sizintisi!"

    features = [f for f in BASE_FEATURES if f not in args.exclude_features]
    df["log_hypo_dist"] = np.log10(np.sqrt(df["distance_km"] ** 2 + df["depth_km"].clip(lower=0) ** 2) + 1)
    if args.with_distance:
        features.append("log_hypo_dist")

    tag = (f"train-{'+'.join(args.train_datasets)}_cap{args.cap}_{args.split_mode}_{args.pool}_"
           f"{'dist' if args.with_distance else 'nodist'}{'_tuned' if args.tune else ''}"
           f"{'_excl-' + '-'.join(args.exclude_features) if args.exclude_features else ''}_seed{args.seed}")
    out_dir = os.path.join(root, "results", tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Deney: {tag} | oznitelik: {len(features)}", flush=True)

    in_tr = df["dataset"].isin(args.train_datasets)
    train = balanced_sample(df[in_tr & (df["split"] == "train")], args.cap, args.seed)
    dev = df[in_tr & (df["split"] == "dev")]
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Egitim: {len(train):,} | Dogrulama: {len(dev):,} | Test: {len(test):,}")
    print(pd.crosstab(train["dataset"], pd.cut(train["magnitude"], MAG_BINS, include_lowest=True)).to_string())

    Xtr, ytr = train[features].values, train["magnitude"].values
    Xdv, ydv = dev[features].values, dev["magnitude"].values
    Xte = test[features].values
    train_keys = set(train["network"].astype(str) + "." + train["station"].astype(str))

    models, params_used, best_iters = {}, {}, {}
    preds_dev, preds_test = {}, {}

    # Referans modeller: klasik Pd olcekleme ve Ridge (tum oznitelikler)
    dist_col = "log_hypo_dist" if args.with_distance else "sp_time_s"
    pd_cols = ["Pd_3s"] + ([dist_col] if dist_col in features else [])
    ref = {
        "PD_SCALING": (make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), LinearRegression()), pd_cols),
        "RIDGE": (make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                RidgeCV(alphas=np.logspace(-3, 3, 13))), features),
    }
    for name, (m, cols) in ref.items():
        m.fit(train[cols].values, ytr)
        models[name] = (m, cols)
        preds_dev[name], preds_test[name] = m.predict(dev[cols].values), m.predict(test[cols].values)
        print(f"  {name}: dev MAE={mean_absolute_error(ydv, preds_dev[name]):.3f}", flush=True)

    # ------------------------------------------------ temel model havuzu
    col = {f: i for i, f in enumerate(features)}
    ALL = list(range(len(features)))
    P_COLS = [col[f] for f in P_FEATS + ["sp_time_s"] if f in col]
    F_COLS = [col[f] for f in FULL_FEATS + SPEC_FEATS + ["sp_time_s"] if f in col]
    imp = lambda: SimpleImputer(strategy="median", keep_empty_features=True)  # noqa: E731
    sk_pool = {"RF": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                           n_jobs=args.n_jobs, random_state=args.seed)}
    if args.pool == "diverse":
        sk_pool["ET"] = make_pipeline(imp(), ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=args.n_jobs,
            random_state=args.seed))
        sk_pool["KNN"] = make_pipeline(imp(), StandardScaler(), KNeighborsRegressor(
            n_neighbors=25, weights="distance", n_jobs=args.n_jobs))
        sk_pool["MLP"] = make_pipeline(imp(), StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
            max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
            random_state=args.seed))

    spec = {}
    for name, est in sk_pool.items():
        t0 = time.time()
        est.fit(Xtr, ytr)
        models[name] = est
        spec[name] = {"kind": "sk", "cols": ALL, "est": est}
        preds_dev[name], preds_test[name] = est.predict(Xdv), est.predict(Xte)
        print(f"  {name}: {time.time() - t0:.0f} s, dev MAE={mean_absolute_error(ydv, preds_dev[name]):.3f}",
              flush=True)

    gbm_jobs = [(n, n, ALL) for n in GBM_NAMES]
    if args.pool == "diverse":
        gbm_jobs += [("LGBM_P", "LGBM", P_COLS), ("LGBM_F", "LGBM", F_COLS)]
    defaults = default_params(args.seed, args.n_jobs)
    for name, gname, cols in gbm_jobs:
        t0 = time.time()
        base = dict(params_used.get(gname, defaults[gname])) if name != gname else dict(defaults[gname])
        if args.tune and name == gname:
            base, best_val = tune_gbm(gname, base, Xtr, ytr, Xdv, ydv, args.tune, args.seed)
            print(f"  {name} Optuna: en iyi dev MAE={best_val:.3f}", flush=True)
        model, best = fit_gbm(gname, build_gbm(gname, base), Xtr[:, cols], ytr, Xdv[:, cols], ydv)
        models[name], params_used[name], best_iters[name] = model, base, best
        spec[name] = {"kind": "gbm", "cols": cols, "gname": gname, "params": base, "best": best}
        preds_dev[name], preds_test[name] = model.predict(Xdv[:, cols]), model.predict(Xte[:, cols])
        print(f"  {name}: {time.time() - t0:.0f} s, agac={best}, "
              f"dev MAE={mean_absolute_error(ydv, preds_dev[name]):.3f}", flush=True)

    stack_names = list(spec)

    # Cesitlilik tanisi: dogrulama setinde hata korelasyonu (dusuk = cesitli)
    R = np.column_stack([preds_dev[n] - ydv for n in stack_names])
    corr = pd.DataFrame(np.corrcoef(R.T), index=stack_names, columns=stack_names)
    corr.to_csv(os.path.join(out_dir, "error_correlation_dev.csv"))
    print("\nHata korelasyonu (dogrulama):")
    print(corr.round(2).to_string())

    if not args.no_stack:
        print(f"Stacking: olay bazli {args.folds} katli GroupKFold, {len(stack_names)} model...", flush=True)
        oof = np.zeros((len(train), len(stack_names)))
        for fi, (a, b) in enumerate(GroupKFold(n_splits=args.folds).split(
                Xtr, ytr, groups=train["event_id"].values)):
            for j, name in enumerate(stack_names):
                sp_ = spec[name]
                cols = sp_["cols"]
                m = clone(sp_["est"]) if sp_["kind"] == "sk" else fixed_size(sp_["gname"], sp_["params"], sp_["best"])
                m.fit(Xtr[a][:, cols], ytr[a])
                oof[b, j] = m.predict(Xtr[b][:, cols])
            print(f"  kat {fi + 1}/{args.folds}", flush=True)
        meta = LinearRegression(positive=True).fit(oof, ytr)
        models["STACK_META"] = meta
        preds_dev["STACK"] = meta.predict(np.column_stack([preds_dev[n] for n in stack_names]))
        preds_test["STACK"] = meta.predict(np.column_stack([preds_test[n] for n in stack_names]))
        print("  Agirliklar:", {n: round(float(c), 3) for n, c in zip(stack_names, meta.coef_)},
              f"sabit={meta.intercept_:.3f}")

    P_dev = np.column_stack([preds_dev[n] for n in stack_names])
    P_test = np.column_stack([preds_test[n] for n in stack_names])
    preds_dev["AVG"], preds_test["AVG"] = P_dev.mean(1), P_test.mean(1)
    blend = LinearRegression(positive=True).fit(P_dev, ydv)   # dogrulama setiyle (test'e dokunmaz)
    models["BLEND_META"] = blend
    preds_dev["BLEND"], preds_test["BLEND"] = blend.predict(P_dev), blend.predict(P_test)

    if not args.no_stack:
        print(f"KNORA-E/U: k={args.des_k}, tol={args.des_tol}", flush=True)
        space = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()).fit(Xtr)
        rng = np.random.default_rng(args.seed)
        dsel = rng.choice(len(Xtr), size=min(args.des_max, len(Xtr)), replace=False)
        Z_dsel, oof_dsel, y_dsel = space.transform(Xtr[dsel]), oof[dsel], ytr[dsel]
        t0 = time.time()
        preds_dev["KNORA_E"], preds_dev["KNORA_U"] = knora_regression(
            Z_dsel, oof_dsel, y_dsel, space.transform(Xdv), P_dev, args.des_k, args.des_tol, args.n_jobs)
        preds_test["KNORA_E"], preds_test["KNORA_U"] = knora_regression(
            Z_dsel, oof_dsel, y_dsel, space.transform(Xte), P_test, args.des_k, args.des_tol, args.n_jobs)
        models["KNORA"] = {"space": space, "Z_dsel": Z_dsel, "oof_dsel": oof_dsel, "y_dsel": y_dsel,
                           "k": args.des_k, "tol": args.des_tol}
        print(f"  KNORA tamam: {time.time() - t0:.0f} s", flush=True)

    # Dogrulama metrikleri (alt orneklem siniri ve model secimi BURADAN yapilir)
    dev_rows = [dict(model=n, **metrics(ydv, p)) for n, p in preds_dev.items()]
    pd.DataFrame(dev_rows).to_csv(os.path.join(out_dir, "metrics_dev.csv"), index=False)

    res, res_bin = evaluate(test, preds_test, train_keys, args.train_datasets, args.seed)
    res.to_csv(os.path.join(out_dir, "metrics_test.csv"), index=False)
    res_bin.to_csv(os.path.join(out_dir, "metrics_test_by_bin.csv"), index=False)
    print("\n=== Test (iz duzeyi, tum izler) ===")
    show = res[(res["level"] == "trace") & (res["subset"] == "all")]
    print(show.drop(columns=["level", "subset"]).round(3).to_string(index=False))
    print("\n=== Test (olay duzeyi, medyan) ===")
    print(res[res["level"] == "event"][["model", "test_set", "n", "MAE", "MAE_lo", "MAE_hi", "bias"]]
          .round(3).to_string(index=False))

    pred_df = test[["dataset", "trace_name", "event_id", "network", "station", "magnitude",
                    "distance_km"]].copy()
    for n, p in preds_test.items():
        pred_df[f"pred_{n}"] = p
    pred_df.to_csv(os.path.join(out_dir, "predictions_test.csv.gz"), index=False)
    imp = pd.DataFrame({"feature": features,
                        "LGBM_gain": models["LGBM"].booster_.feature_importance("gain")})
    imp.sort_values("LGBM_gain", ascending=False).to_csv(
        os.path.join(out_dir, "feature_importance.csv"), index=False)
    spec_min = {n: {k: v for k, v in d.items() if k not in ("est", "params")} for n, d in spec.items()}
    joblib.dump({"models": models, "features": features, "stack_names": stack_names, "spec": spec_min,
                 "args": vars(args)}, os.path.join(out_dir, "models.joblib"))
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({"features": features, "best_iterations": best_iters,
                   "params": {k: {kk: vv for kk, vv in v.items() if kk not in ("n_jobs", "thread_count")}
                              for k, v in params_used.items()}, **vars(args)}, fh, indent=2, default=str)
    print(f"\nSonuclar: {out_dir}")


if __name__ == "__main__":
    main()