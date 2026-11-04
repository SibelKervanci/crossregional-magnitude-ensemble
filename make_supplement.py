"""
Ek materyal tablolari (Spyder: F5). Yeni hesaplama yapmaz; mevcut sonuc dosyalarini toplar.

Uretilenler (DataBase/results/paper/supplement/):
  S2_knora_grid.tex / .csv       : bolgeler arasi ayarlamadaki tum KNORA ayarlari + referans yontemler
  S3_full_metrics.tex / .csv     : tum degerlendirme setleri x yontemler (yonlendirilmis)
  S4_ablations.tex / .csv        : ablasyonlar (bolge ici test setleri; varsa Turkiye)
  S5_hyperparameters.tex / .csv  : modellerin kayitli parametreleri (models.joblib'den okunur)

Once make_results.py ve tune_ensembles.py calismis olmali. Ablasyonlarin Turkiye sutunu
icin evaluate_turkey.py ablasyonlar bittikten sonra bir kez daha calistirilmis olmali (istege bagli).
supplementary.tex bu dosyalari \\input ile icerir.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import cfg  # noqa: E402

ROOT, SEED = cfg("ROOT"), cfg("SEED", 42)
DATASETS = cfg("DATASETS", ["STEAD", "INSTANCE", "CEED"])
ALL = "+".join(DATASETS)
MAIN = f"train-{ALL}_cap40000_event_diverse_nodist_seed{SEED}"
ABLATIONS = {
    "Main (balanced, diverse pool)": MAIN,
    "Natural magnitude distribution$^a$": f"train-{ALL}_cap0_event_diverse_nodist_seed{SEED}",
    "With catalogue distance": f"train-{ALL}_cap40000_event_diverse_dist_seed{SEED}",
    "Tree-based pool only": f"train-{ALL}_cap40000_event_trees_nodist_seed{SEED}",
    "Time-based split": f"train-{ALL}_cap40000_time_diverse_nodist_seed{SEED}",
}
OUT = os.path.join(ROOT, "results", "paper", "supplement")
LABEL = {"PD_SCALING": "$P_d$ scaling", "RIDGE": "Ridge", "KNORA_E": "KNORA-E", "KNORA_U": "KNORA-U",
         "LGBM": "LightGBM", "CAT": "CatBoost", "XGB": "XGBoost", "STACK": "Stacking", "AVG": "Average",
         "BLEND": "Blending", "LGBM_P": "LGBM-P", "LGBM_F": "LGBM-F"}


def tex_escape(x):
    """Metni LaTeX icin kacirir; $...$ icindeki matematik kisimlarina dokunmaz."""
    parts = str(x).split("$")
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:                                  # matematik kismi
            out.append("$" + part + "$")
            continue
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
            part = part.replace(a, b)
        out.append(part.replace("->", r"$\to$").replace("Turkiye", r"T\"urkiye"))
    return "".join(out)


def longtable(df, path, caption, label, colspec=None, note=None):
    """Basit longtable yazici (pandas/jinja2 gerektirmez)."""
    cols = list(df.columns)
    colspec = colspec or "l" * len(cols)
    head = " & ".join(tex_escape(c) for c in cols) + r" \\"
    lines = [r"\begin{longtable}{" + colspec + "}",
             r"\caption{" + caption + r"}\label{" + label + r"}\\",
             r"\toprule", head, r"\midrule", r"\endfirsthead",
             r"\toprule", head, r"\midrule", r"\endhead",
             r"\bottomrule", r"\endfoot"]
    for _, r in df.iterrows():
        lines.append(" & ".join(tex_escape(v) for v in r.values) + r" \\")
    lines.append(r"\end{longtable}")
    if note:
        lines.append(r"\noindent{\footnotesize " + note + r"}\par\medskip")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    df.to_csv(path.replace(".tex", ".csv"), index=False)
    print(f"  {os.path.basename(path)}: {len(df)} satir")


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


# ------------------------------------------------------------------ S2
def table_s2():
    p = os.path.join(ROOT, "results", "ensemble_tuning_routed", "summary.csv")
    if not os.path.exists(p):
        print("  S2: ensemble_tuning_routed/summary.csv yok, atlandi")
        return
    s = pd.read_csv(p)
    s.columns = ["method", "mean", "worst"]
    rows = []
    for _, r in s.iterrows():
        m = r["method"]
        if m.startswith("KNORA_"):
            kind, space, k, par = m.split("|")
            rows.append({"Method": "KNORA-" + kind.split("_")[1], "Competence space": space.split("=")[1],
                         "$k$": k.split("=")[1],
                         "Tolerance / exponent": par.split("=")[1],
                         "Mean MAE": fmt(r["mean"], 4), "Worst-case MAE": fmt(r["worst"], 4)})
        else:
            rows.append({"Method": LABEL.get(m, m), "Competence space": "--", "$k$": "--",
                         "Tolerance / exponent": "--", "Mean MAE": fmt(r["mean"], 4),
                         "Worst-case MAE": fmt(r["worst"], 4)})
    df = pd.DataFrame(rows)
    longtable(df, os.path.join(OUT, "S2_knora_grid.tex"),
              "Leave-one-domain-out validation of ensemble settings (routed predictions). Mean and worst-case "
              "MAE over the three source--target configurations; sorted by mean MAE.",
              "tab:s2", "llrrrr",
              note="Competence space ``phys'': S--P time, PGV, PGD, PGA, $D_{5\\text{--}95}$, dominant frequency, "
                   "spectral centroid, $P_d$ and $\\tau_c$ at 3~s. KNORA-E: tolerance $\\epsilon$; KNORA-U: weighting exponent $p$.")


# ------------------------------------------------------------------ S3
def table_s3():
    p = os.path.join(ROOT, "results", "paper", "table_main.csv")
    if not os.path.exists(p):
        print("  S3: table_main.csv yok (once make_results.py), atlandi")
        return
    t = pd.read_csv(p)
    t["set"] = t["set"].str.replace("All->", "All three->", regex=False)
    df = pd.DataFrame({
        "Training $\\to$ test": t["set"].str.replace("->", " -> ", regex=False),
        "Method": t["method"].map(lambda m: LABEL.get(m, m)),
        "$n$": t["n"].map(lambda v: f"{int(v):,}"),
        "MAE [95% CI]": t.apply(lambda r: f"{r.MAE:.3f} [{r.MAE_lo:.3f}, {r.MAE_hi:.3f}]", axis=1),
        "RMSE": t["RMSE"].map(fmt), "$R^2$": t["R2"].map(fmt), "Bias": t["bias"].map(lambda v: f"{v:+.3f}"),
        "$\\sigma$": t["sigma"].map(fmt),
        "$|e|\\le0.3$": t["within_0.3"].map(lambda v: f"{100 * v:.1f}%"),
        "$|e|\\le0.5$": t["within_0.5"].map(lambda v: f"{100 * v:.1f}%"),
    })
    longtable(df, os.path.join(OUT, "S3_full_metrics.tex"),
              "Trace-level metrics for all methods and evaluation sets (routed predictions). Confidence "
              "intervals from 1000 event-level bootstrap resamples.", "tab:s3", "llrlrrrrrr")


# ------------------------------------------------------------------ S4
def table_s4():
    rows = []
    tr = None
    tp = os.path.join(ROOT, "results", "turkey_eval", "metrics_turkey.csv")
    if os.path.exists(tp):
        tr = pd.read_csv(tp)
        tr = tr[(tr["subset"] == "all") & (tr["level"] == "trace")]
    models = ["PD_SCALING", "CAT", "MLP", "AVG", "KNORA_E", "STACK"]
    for name, run in ABLATIONS.items():
        p = os.path.join(ROOT, "results", run, "metrics_test.csv")
        if not os.path.exists(p):
            print(f"  S4: {run} yok, atlandi")
            continue
        m = pd.read_csv(p)
        if "subset" in m.columns:                       # yalnizca tum izler
            m = m[m["subset"].astype(str).isin(["all", "nan"])]
        tests = [(ds, m[m["test_set"] == ds].drop_duplicates("model").set_index("model")) for ds in DATASETS]
        if tr is not None and (tr["run"] == run).any():
            tests.append(("Turkiye", tr[tr["run"] == run].drop_duplicates("model").set_index("model")))
        for ds, g in tests:
            row = {"Configuration": name, "Test set": ds}
            for mod in models:
                row[LABEL.get(mod, mod)] = (f"{g.loc[mod, 'MAE']:.3f} ({g.loc[mod, 'bias']:+.2f})"
                                            if mod in g.index else "--")
            rows.append(row)
    if rows:
        longtable(pd.DataFrame(rows), os.path.join(OUT, "S4_ablations.tex"),
                  "Ablations, trace-level MAE (bias in parentheses) of full models trained on all three "
                  "benchmarks. Ablation runs have no S--P-free counterpart, so predictions are not routed; "
                  "routing affects only records without an S arrival (12.7\\% of INSTANCE, 38\\% of T\\\"urkiye).",
                  "tab:s4", "ll" + "r" * len(models),
                  note="$^a$ Trained on 1.1 million traces without stacking and KNORA; ``--'': not available.")


# ------------------------------------------------------------------ S5
KEEP = {"n_estimators", "max_depth", "min_samples_leaf", "max_features", "n_neighbors", "weights",
        "hidden_layer_sizes", "alpha", "batch_size", "learning_rate_init", "max_iter", "early_stopping",
        "validation_fraction", "n_iter_no_change", "learning_rate", "subsample", "colsample_bytree",
        "min_child_weight", "reg_lambda", "num_leaves", "min_child_samples", "depth", "l2_leaf_reg",
        "iterations", "tree_method", "alphas"}


def table_s5():
    p = os.path.join(ROOT, "results", MAIN, "models.joblib")
    if not os.path.exists(p):
        print("  S5: models.joblib yok, atlandi")
        return
    b = joblib.load(p)
    cfgp = os.path.join(ROOT, "results", MAIN, "config.json")
    c = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    best_iters = c.get("best_iterations", {})
    rows = []
    for name, obj in b["models"].items():
        est = obj[0] if isinstance(obj, tuple) else obj
        if isinstance(est, dict) or name in ("STACK_META", "BLEND_META"):
            continue
        final = est.steps[-1][1] if hasattr(est, "steps") else est
        try:
            params = final.get_params()
        except Exception:
            continue
        if hasattr(final, "get_all_params"):           # CatBoost
            params = {**final.get_params()}
        sel = {}
        for k, v in params.items():
            if k not in KEEP or v is None:
                continue
            if isinstance(v, np.ndarray):                   # or. RidgeCV alpha izgarasi
                v = f"{v.min():g} to {v.max():g} ({len(v)} log-spaced values)"
            sel[k] = v
        if name == "PD_SCALING":
            sel = {"model": "ordinary least squares on log Pd (3 s) and S-P time"}
        best = best_iters.get(name)
        if best:
            sel["trees used (early stopping)"] = best
        rows.append({"Model": LABEL.get(name, name), "Library": type(final).__module__.split(".")[0],
                     "Parameters": "; ".join(f"{k} = {v}" for k, v in sorted(sel.items()))})
    kn = b["models"].get("KNORA", {})
    rows.append({"Model": "KNORA-E / KNORA-U", "Library": "own implementation",
                 "Parameters": f"k = {kn.get('k')}; tolerance = {kn.get('tol')}; competence set = "
                               f"{len(kn.get('y_dsel', []))} out-of-fold training records"})
    rows.append({"Model": "Stacking", "Library": "sklearn",
                 "Parameters": "non-negative linear regression on 5-fold event-grouped out-of-fold predictions"})
    if c:
        a = c
        rows.append({"Model": "General", "Library": "--",
                     "Parameters": f"training cap per benchmark and magnitude bin = {a.get('cap')}; "
                                   f"SNR threshold = {a.get('min_snr')} dB; seed = {a.get('seed')}; "
                                   f"band-pass 1--20 Hz"})
    longtable(pd.DataFrame(rows), os.path.join(OUT, "S5_hyperparameters.tex"),
              "Hyperparameters of the models trained on all three benchmarks (identical in all experiments; "
              "read from the saved models).", "tab:s5", "llp{10cm}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Ek materyal -> {OUT}")
    table_s2()
    table_s3()
    table_s4()
    table_s5()


if __name__ == "__main__":
    main()