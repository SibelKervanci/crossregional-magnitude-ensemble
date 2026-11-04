"""
Makale icin tablolar, istatistiksel testler ve sekiller (Spyder: F5).
Tum sonuclar YONLENDIRILMIS tahminlerle (S varsa tam model, yoksa S-P'siz model) hesaplanir.

Uretilenler (DataBase/results/paper/):
  table_main.csv / .tex     : tum degerlendirme setleri x yontemler (MAE [%95 GA], RMSE, R2, sapma,
                              sigma, |e|<=0.3 ve |e|<=0.5 orani)
  paired_tests.csv          : STACK - diger yontem (olay bazli bootstrap dMAE, Wilcoxon, Holm)
  friedman.txt              : Friedman testi + Nemenyi kritik fark
  magnitude_matched.csv     : bolge ici MAE, Turkiye buyukluk dagilimina gore agirliklandirilmis
  Fig_pred_obs, Fig_transfer_matrix, Fig_missingness, Fig_turkey_bins,
  Fig_diversity, Fig_cd_diagram, Fig_fewshot, Fig_magnitude_matched  (.pdf + .tif 600 dpi)

Sekil standartlari (Elsevier): genislik 90 / 140 / 190 mm; Arial/Helvetica 7-9 pt;
renk korlugune uygun Okabe-Ito paleti.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import glob  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import cfg  # noqa: E402

ROOT, KOERI, SEED = cfg("ROOT"), cfg("KOERI"), cfg("SEED", 42)
DATASETS = cfg("DATASETS", ["STEAD", "INSTANCE", "CEED"])
TAG = f"_cap40000_event_diverse_nodist_seed{SEED}"
NOSP = "_excl-sp_time_s"
MAIN = "train-" + "+".join(DATASETS) + TAG
SINGLE = {d: f"train-{d}{TAG}" for d in DATASETS}
METHODS = ["PD_SCALING", "RIDGE", "RF", "ET", "KNN", "MLP", "XGB", "LGBM", "CAT",
           "AVG", "BLEND", "KNORA_E", "KNORA_U", "STACK"]
LABEL = {"PD_SCALING": "Pd scaling", "RIDGE": "Ridge", "KNORA_E": "KNORA-E", "KNORA_U": "KNORA-U",
         "LGBM": "LightGBM", "CAT": "CatBoost", "XGB": "XGBoost", "STACK": "Stacking",
         "AVG": "Average", "BLEND": "Blending"}
OUT = os.path.join(ROOT, "results", "paper")
N_BOOT = 1000

# ------------------------------------------------------------------ stil
OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "axes.prop_cycle": plt.cycler(color=OKABE), "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def save(fig, name):
    for ext, kw in (("pdf", {}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), **kw)
    plt.close(fig)
    print(f"  sekil: {name}")


def lab(m):
    return LABEL.get(m, m)


# ------------------------------------------------------------------ veri
def has_s_map():
    f = pd.read_csv(os.path.join(ROOT, "features", "features_all.csv.gz"),
                    usecols=["dataset", "trace_name", "sp_time_s"])
    f["has_S"] = f["sp_time_s"].notna()
    return f.drop(columns="sp_time_s")


def routed_test(run, hs):
    """Bolge ici/bolgeler arasi test tahminleri: tam + S-P'siz -> yonlendirilmis."""
    full_p = os.path.join(ROOT, "results", run, "predictions_test.csv.gz")
    nosp_p = os.path.join(ROOT, "results", run.replace("_seed", f"{NOSP}_seed"), "predictions_test.csv.gz")
    if not (os.path.exists(full_p) and os.path.exists(nosp_p)):
        return None
    a, b = pd.read_csv(full_p, low_memory=False), pd.read_csv(nosp_p, low_memory=False)
    d = a.merge(b[["dataset", "trace_name"] + [c for c in b if c.startswith("pred_")]],
                on=["dataset", "trace_name"], suffixes=("_full", "_nosp")).merge(hs, on=["dataset", "trace_name"])
    s = d["has_S"].values.astype(bool)
    out = d[["dataset", "trace_name", "event_id", "magnitude", "distance_km", "has_S"]].copy()
    for m in METHODS:
        if f"pred_{m}_full" in d and f"pred_{m}_nosp" in d:
            out[f"pred_{m}"] = np.where(s, d[f"pred_{m}_full"], d[f"pred_{m}_nosp"])
            out[f"full_{m}"], out[f"nosp_{m}"] = d[f"pred_{m}_full"], d[f"pred_{m}_nosp"]
    return out


def turkey(run, kind="ROUTED"):
    lab_run = run.replace("_seed", f"_{kind}_seed") if kind else run
    p = os.path.join(ROOT, "results", "turkey_eval", f"pred_{lab_run}.csv.gz")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, dtype={"event_id": str})
    tf = pd.read_csv(os.path.join(KOERI, "turkey_features.csv.gz"), dtype={"event_id": str},
                     usecols=["event_id", "station", "has_S"])
    d = d.merge(tf, on=["event_id", "station"], how="left").rename(columns={"ML_bulletin": "magnitude"})
    return d


# ------------------------------------------------------------------ buyukluk esleme
MAG_BINS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.5]


def mag_matched(d_in, d_tr, m, rng):
    """Bolge ici test izlerini Turkiye'nin buyukluk dagilimina gore agirliklandirilmis MAE.
    Yalnizca her iki sette de en az 20 iz bulunan araliklar kullanilir."""
    bi = pd.cut(d_in["magnitude"], MAG_BINS)
    bt = pd.cut(d_tr["magnitude"], MAG_BINS)
    ni, nt = bi.value_counts(), bt.value_counts()
    ok = [b for b in ni.index if ni[b] >= 20 and nt.get(b, 0) >= 20]
    pt = nt[ok] / nt[ok].sum()
    coverage = nt[ok].sum() / len(d_tr)
    keep = bi.isin(ok).values
    w = (bi[keep].map(pt) / bi[keep].map(ni[ok] / ni[ok].sum())).astype(float).values
    ae = np.abs(d_in[f"pred_{m}"].values[keep] - d_in["magnitude"].values[keep])
    ev = d_in["event_id"].values[keep]
    g = pd.DataFrame({"e": ev, "wa": w * ae, "w": w}).groupby("e")[["wa", "w"]].sum()
    a, b = g["wa"].values, g["w"].values
    idx = rng.integers(0, len(a), (N_BOOT, len(a)))
    boot = a[idx].sum(1) / b[idx].sum(1)
    keep_t = bt.isin(ok).values
    mae_t = np.mean(np.abs(d_tr[f"pred_{m}"].values[keep_t] - d_tr["magnitude"].values[keep_t]))
    return {"MAE_in_weighted": a.sum() / b.sum(), "lo": np.percentile(boot, 2.5),
            "hi": np.percentile(boot, 97.5), "MAE_turkiye_same_bins": mae_t, "turkiye_coverage": coverage}


# ------------------------------------------------------------------ istatistik
def event_boot_ci(y, p, ev, rng):
    g = pd.DataFrame({"e": ev, "ae": np.abs(p - y)}).groupby("e")["ae"].agg(["sum", "size"])
    s, n = g["sum"].values, g["size"].values
    idx = rng.integers(0, len(n), (N_BOOT, len(n)))
    return np.percentile(s[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])


def metric_row(y, p, ev, rng):
    e = p - y
    lo, hi = event_boot_ci(y, p, ev, rng)
    return {"n": len(y), "MAE": np.mean(np.abs(e)), "MAE_lo": lo, "MAE_hi": hi,
            "RMSE": np.sqrt(np.mean(e ** 2)), "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
            "bias": e.mean(), "sigma": e.std(ddof=1),
            "within_0.3": np.mean(np.abs(e) <= 0.3), "within_0.5": np.mean(np.abs(e) <= 0.5)}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired_tests(d, set_name, rng, ref="STACK"):
    """Olay duzeyinde: |hata| ortalamasi (olay basina) uzerinden eslestirilmis karsilastirma."""
    meths = [m for m in METHODS if f"pred_{m}" in d]
    ae = pd.DataFrame({m: np.abs(d[f"pred_{m}"].values - d["magnitude"].values) for m in meths})
    ae["event_id"] = d["event_id"].values
    ev = ae.groupby("event_id").mean()
    rows = []
    ae_ref = ev[ref]
    for m in meths:
        if m == ref:
            continue
        ae_m = ev[m]
        diff = (ae_ref - ae_m).values
        idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
        boot = diff[idx].mean(1)
        w = stats.wilcoxon(ae_ref.values, ae_m.values, zero_method="wilcox")
        rows.append({"set": set_name, "method": m, "dMAE_event": diff.mean(),
                     "dMAE_lo": np.percentile(boot, 2.5), "dMAE_hi": np.percentile(boot, 97.5),
                     "wilcoxon_p": w.pvalue, "n_events": len(diff)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["wilcoxon_p"])
    return out


# Nemenyi kritik degerleri (alpha = 0.05), Demsar (2006), Tablo 5a
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
       10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354}


def cd_diagram(avg_ranks, cd, n_sets):
    names = list(avg_ranks.index)
    k = len(names)
    fig, ax = plt.subplots(figsize=(140 * MM, 45 * MM))
    ax.set_xlim(1, k)
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.axis("off")
    ax.hlines(0.85, 1, k, color="k", lw=0.8)
    for r in range(1, k + 1):
        ax.vlines(r, 0.83, 0.87, color="k", lw=0.6)
        ax.text(r, 0.92, str(r), ha="center", fontsize=7)
    order = avg_ranks.sort_values()
    for i, (n, r) in enumerate(order.items()):
        left = i < k / 2
        y = 0.75 - (i if left else k - 1 - i) * 0.09
        x_text = 0.8 if left else k + 0.2
        ax.plot([r, r, x_text], [0.85, y, y], color="k", lw=0.6)
        ax.text(x_text - 0.05 if left else x_text + 0.05, y, f"{lab(n)} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=7)
    ax.plot([1, 1 + cd], [0.98, 0.98], color=OKABE[6], lw=1.5)
    ax.text(1 + cd / 2, 1.0, f"CD = {cd:.2f}", ha="center", fontsize=7, color=OKABE[6])
    ax.set_title(f"Average ranks over {n_sets} evaluation sets (lower is better)", fontsize=8, y=1.05)
    return fig


# ------------------------------------------------------------------ ana akis
def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    hs = has_s_map()

    # Degerlendirme setleri: (kisa ad, egitim, test, DataFrame)
    sets = []
    rt = routed_test(MAIN, hs)
    if rt is not None:
        for ds in DATASETS:
            if (rt["dataset"] == ds).sum() >= 20:
                sets.append((f"All->{ds}", "All", ds, rt[rt["dataset"] == ds]))
    for src, run in SINGLE.items():
        r = routed_test(run, hs)
        if r is None:
            continue
        for ds in DATASETS:          # kendi test seti (kosegen) dahil
            sub = r[r["dataset"] == ds]
            if len(sub) >= 20:
                sets.append((f"{src}->{ds}", src, ds, sub))
    for src, run in [("All", MAIN)] + list(SINGLE.items()):
        t = turkey(run)
        if t is not None:
            sets.append((f"{src}->Turkiye", src, "Turkiye", t))
    print(f"Degerlendirme seti: {len(sets)}")

    # ---- Tablo 1
    rows = []
    for name, src, tgt, d in sets:
        for m in METHODS:
            if f"pred_{m}" in d:
                r = metric_row(d["magnitude"].values, d[f"pred_{m}"].values, d["event_id"].values, rng)
                rows.append({"set": name, "train": src, "test": tgt, "method": m, **r})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "table_main.csv"), index=False)
    compact = tab.assign(cell=tab.apply(lambda r: f"{r.MAE:.3f} [{r.MAE_lo:.3f}, {r.MAE_hi:.3f}]", axis=1))
    piv = compact.pivot(index="set", columns="method", values="cell")
    piv = piv[[m for m in ["PD_SCALING", "RIDGE", "XGB", "CAT", "MLP", "AVG", "KNORA_E", "STACK"] if m in piv]]
    piv.columns = [lab(c) for c in piv.columns]
    with open(os.path.join(OUT, "table_main.tex"), "w") as fh:
        fh.write(piv.to_latex(escape=False))
    print("  tablo: table_main.csv / .tex")

    # ---- Eslestirilmis testler (STACK'e gore)
    pt = pd.concat([paired_tests(d, name, rng) for name, _, _, d in sets
                    if "pred_STACK" in d], ignore_index=True)
    pt.to_csv(os.path.join(OUT, "paired_tests.csv"), index=False)
    print("  tablo: paired_tests.csv")

    # ---- Friedman + Nemenyi
    mae = tab.pivot(index="set", columns="method", values="MAE")
    mae = mae[[m for m in METHODS if m in mae and m != "BLEND"]].dropna()
    if len(mae) >= 3 and mae.shape[1] >= 3:
        fr = stats.friedmanchisquare(*[mae[c].values for c in mae])
        ranks = mae.rank(axis=1).mean()
        k, n = mae.shape[1], mae.shape[0]
        cd = Q05.get(k, 3.4) * np.sqrt(k * (k + 1) / (6 * n))
        with open(os.path.join(OUT, "friedman.txt"), "w") as fh:
            fh.write(f"Friedman chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.2e}, k = {k}, N = {n}\n")
            fh.write(f"Nemenyi CD (alpha=0.05) = {cd:.3f}\n\nOrtalama siralar:\n")
            fh.write(ranks.sort_values().round(3).to_string())
        save(cd_diagram(ranks, cd, n), "Fig_cd_diagram")

    # ---- Buyukluge gore eslestirilmis karsilastirma (bolge ici vs Turkiye)
    tr_all = next((d for name, src, tgt, d in sets if src == "All" and tgt == "Turkiye"), None)
    if tr_all is not None:
        rows = []
        for name, src, tgt, d in sets:
            if src == "All" and tgt != "Turkiye":
                for m in ("STACK", "CAT", "MLP", "PD_SCALING"):
                    rows.append({"in_region_set": tgt, "method": m, **mag_matched(d, tr_all, m, rng)})
        mm = pd.DataFrame(rows)
        mm.to_csv(os.path.join(OUT, "magnitude_matched.csv"), index=False)
        print("  tablo: magnitude_matched.csv")
        print(mm[mm["method"] == "STACK"].round(3).to_string(index=False))

        fig, ax = plt.subplots(figsize=(140 * MM, 62 * MM))
        groups = [(tgt, d) for name, src, tgt, d in sets if src == "All"]
        for (tgt, d), c, mk in zip(groups, [OKABE[5], OKABE[1], OKABE[3], OKABE[6]], "osD^"):
            d = d.assign(bin=pd.cut(d["magnitude"], MAG_BINS))
            xs, mae, lo, hi = [], [], [], []
            for b, gg in d.groupby("bin", observed=True):
                if len(gg) < 20:
                    continue
                r = metric_row(gg["magnitude"].values, gg["pred_STACK"].values, gg["event_id"].values, rng)
                xs.append(b.mid)
                mae.append(r["MAE"])
                lo.append(r["MAE_lo"])
                hi.append(r["MAE_hi"])
            ax.errorbar(xs, mae, yerr=[np.subtract(mae, lo), np.subtract(hi, mae)], color=c, marker=mk,
                        ms=3.5, capsize=2, label=tgt.replace("Turkiye", "Türkiye (zero-shot)"))
        ax.set_xlabel("Catalogue $M_L$ (0.5-unit bins)")
        ax.set_ylabel("MAE, stacking")
        ax.legend(frameon=False)
        ax.grid(lw=0.3, alpha=0.5)
        save(fig, "Fig_magnitude_matched")

    # ---- Sekil: tahmin-gozlem (ana deney, bolge ici + Turkiye)
    panels = [s for s in sets if s[1] == "All"]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(190 * MM, 50 * MM), sharex=True, sharey=True)
        for ax, (name, _, tgt, d) in zip(np.atleast_1d(axes), panels):
            hb = ax.hexbin(d["magnitude"], d["pred_STACK"], gridsize=45, bins="log", cmap="cividis",
                           mincnt=1, extent=(0, 6.5, 0, 6.5), linewidths=0)
            ax.plot([0, 6.5], [0, 6.5], "k-", lw=0.6)
            for off in (-0.5, 0.5):
                ax.plot([0, 6.5], [off, 6.5 + off], "k--", lw=0.4)
            r = tab[(tab["set"] == name) & (tab["method"] == "STACK")].iloc[0]
            ax.text(0.05, 0.95, f"{tgt.replace('Turkiye', 'Türkiye')}\nMAE = {r.MAE:.3f}\n"
                    f"σ = {r.sigma:.3f}\nbias = {r.bias:+.3f}", transform=ax.transAxes, va="top", fontsize=6.5)
            ax.set_xlabel("Catalogue $M_L$")
            ax.set_aspect("equal")
        np.atleast_1d(axes)[0].set_ylabel("Predicted $M_L$ (stacking)")
        fig.colorbar(hb, ax=axes, shrink=0.8, label="Traces (log)")
        save(fig, "Fig_pred_obs")

    # ---- Sekil: aktarim matrisi (STACK ve Pd olcekleme)
    trains = ["All"] + list(SINGLE)
    tests = DATASETS + ["Turkiye"]
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM))
    for ax, m in zip(axes, ["STACK", "PD_SCALING"]):
        M = np.full((len(trains), len(tests)), np.nan)
        for i, tr in enumerate(trains):
            for j, te in enumerate(tests):
                q = tab[(tab["train"] == tr) & (tab["test"] == te) & (tab["method"] == m)]
                if len(q):
                    M[i, j] = q["MAE"].iloc[0]
        im = ax.imshow(M, cmap="viridis_r", vmin=0.15, vmax=0.70)
        for i in range(len(trains)):
            for j in range(len(tests)):
                if not np.isnan(M[i, j]):
                    in_dom = trains[i] == tests[j] or (trains[i] == "All" and tests[j] != "Turkiye")
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                            color="w" if M[i, j] > 0.45 else "k", fontweight="bold" if in_dom else "normal")
        ax.set_xticks(range(len(tests)), [t.replace("Turkiye", "Türkiye") for t in tests])
        ax.set_yticks(range(len(trains)), ["All three"] + trains[1:])
        ax.set_xlabel("Test set")
        ax.set_ylabel("Training set" if m == "STACK" else "")
        ax.set_title(lab(m))
    fig.colorbar(im, ax=axes, shrink=0.85, label="MAE (magnitude units)")
    save(fig, "Fig_transfer_matrix")

    # ---- Sekil: eksiklik kaymasi (S varisi olmayan izlerde sapma)
    rows = []
    for name, src, tgt, d in sets:
        if tgt == "Turkiye":
            full = turkey(MAIN if src == "All" else SINGLE[src], kind=None)
            nosp = turkey((MAIN if src == "All" else SINGLE[src]).replace("_seed", f"{NOSP}_seed"), kind=None)
            if full is None or nosp is None:
                continue
            for var, dd in (("full", full), ("S-P-free", nosp), ("routed", d)):
                k = ~dd["has_S"].astype(bool)
                for m in ("STACK", "CAT"):
                    e = dd.loc[k, f"pred_{m}"] - dd.loc[k, "magnitude"]
                    rows.append({"set": name, "variant": var, "method": m, "bias": e.mean(), "MAE": e.abs().mean()})
        elif "full_STACK" in d and (~d["has_S"]).sum() >= 50:
            k = ~d["has_S"].values
            for m in ("STACK", "CAT"):
                for var, col in (("full", f"full_{m}"), ("S-P-free", f"nosp_{m}"), ("routed", f"pred_{m}")):
                    e = d.loc[k, col] - d.loc[k, "magnitude"]
                    rows.append({"set": name, "variant": var, "method": m, "bias": e.mean(), "MAE": e.abs().mean()})
    miss = pd.DataFrame(rows)
    if len(miss):
        miss.to_csv(os.path.join(OUT, "missingness.csv"), index=False)
        fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 60 * MM), sharey=True)
        for ax, m in zip(axes, ("STACK", "CAT")):
            q = miss[miss["method"] == m]
            names = list(dict.fromkeys(q["set"]))
            x = np.arange(len(names))
            for i, (var, c, mk) in enumerate((("full", OKABE[6], "o"), ("S-P-free", OKABE[2], "s"),
                                               ("routed", OKABE[3], "D"))):
                v = q[q["variant"] == var].set_index("set").reindex(names)["bias"]
                ax.scatter(x + (i - 1) * 0.22, v, color=c, marker=mk, s=14, label=var, zorder=3)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x, [n.replace("->", "→").replace("Turkiye", "Türkiye") for n in names],
                          rotation=35, ha="right")
            ax.set_title(f"{lab(m)}: traces without S arrival")
            ax.grid(axis="y", lw=0.3, alpha=0.5)
        axes[0].set_ylabel("Mean bias (predicted − catalogue)")
        axes[0].legend(frameon=False, loc="lower left")
        save(fig, "Fig_missingness")

    # ---- Sekil: Turkiye buyukluk ve uzaklik araliklari
    t = turkey(MAIN)
    if t is not None:
        fig, axes = plt.subplots(2, 2, figsize=(190 * MM, 95 * MM), sharex="col")
        for col, (var, bins, xl) in enumerate((("magnitude", [2, 2.5, 3, 3.5, 4, 4.5, 5, 7], "Catalogue $M_L$"),
                                                ("distance_km", [0, 20, 40, 60, 80, 110], "Epicentral distance (km)"))):
            t["bin"] = pd.cut(t[var], bins)
            for m, c in (("STACK", OKABE[5]), ("CAT", OKABE[1]), ("PD_SCALING", OKABE[0])):
                g = t.groupby("bin", observed=True)
                xs, mae, lo, hi, bias = [], [], [], [], []
                for b, gg in g:
                    if len(gg) < 30:
                        continue
                    r = metric_row(gg["magnitude"].values, gg[f"pred_{m}"].values, gg["event_id"].values, rng)
                    xs.append(b.mid)
                    mae.append(r["MAE"])
                    lo.append(r["MAE_lo"])
                    hi.append(r["MAE_hi"])
                    bias.append(r["bias"])
                axes[0, col].errorbar(xs, mae, yerr=[np.subtract(mae, lo), np.subtract(hi, mae)],
                                      color=c, marker="o", ms=3, capsize=2, label=lab(m))
                axes[1, col].plot(xs, bias, color=c, marker="o", ms=3)
            axes[1, col].axhline(0, color="k", lw=0.5)
            axes[1, col].set_xlabel(xl)
        axes[0, 0].set_ylabel("MAE")
        axes[1, 0].set_ylabel("Bias")
        axes[0, 0].legend(frameon=False)
        for ax in axes.ravel():
            ax.grid(lw=0.3, alpha=0.5)
        save(fig, "Fig_turkey_bins")

    # ---- Sekil: cesitlilik (hata korelasyonu + stacking agirliklari)
    corr_p = os.path.join(ROOT, "results", MAIN, "error_correlation_dev.csv")
    if os.path.exists(corr_p):
        C = pd.read_csv(corr_p, index_col=0)
        fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 70 * MM), gridspec_kw={"width_ratios": [1.3, 1]})
        im = axes[0].imshow(C.values, cmap="cividis", vmin=0.6, vmax=1.0)
        names = [lab(c) for c in C.columns]
        axes[0].set_xticks(range(len(names)), names, rotation=45, ha="right")
        axes[0].set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                axes[0].text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                             color="k" if C.values[i, j] > 0.85 else "w")
        fig.colorbar(im, ax=axes[0], shrink=0.8, label="Error correlation (validation)")
        b = joblib.load(os.path.join(ROOT, "results", MAIN, "models.joblib"))
        w = pd.Series(b["models"]["STACK_META"].coef_, index=b["stack_names"])
        axes[1].barh([lab(n) for n in w.index], w.values, color=OKABE[5])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Stacking weight")
        axes[1].grid(axis="x", lw=0.3, alpha=0.5)
        save(fig, "Fig_diversity")

    # ---- Sekil: az ornekli sapma duzeltmesi
    fp = os.path.join(ROOT, "results", "turkey_eval", "fewshot_turkey.csv")
    if os.path.exists(fp):
        fs = pd.read_csv(fp)
        fs = fs[fs["run"] == MAIN.replace("_seed", "_ROUTED_seed")]
        fig, ax = plt.subplots(figsize=(90 * MM, 60 * MM))
        for m, c in zip(["STACK", "MLP", "CAT", "KNORA_E", "PD_SCALING"], [OKABE[5], OKABE[3], OKABE[1],
                                                                           OKABE[7], OKABE[0]]):
            q = fs[fs["model"] == m].sort_values("n_calib")
            if len(q):
                ax.plot(q["n_calib"], q["MAE"], marker="o", ms=3, color=c, label=lab(m))
        ax.set_xlabel("Calibration events $n$ (0 = zero-shot)")
        ax.set_ylabel("Event-level MAE (later events)")
        ax.legend(frameon=False, fontsize=6.5)
        ax.grid(lw=0.3, alpha=0.5)
        save(fig, "Fig_fewshot")

    print(f"\nKaydedildi: {OUT}")


if __name__ == "__main__":
    main()