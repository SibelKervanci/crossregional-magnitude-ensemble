"""
SHAP analizi: hangi fiziksel oznitelikler bolgeler arasinda tasinabiliyor? (Spyder: F5)

Temsilci model: LightGBM (tam oznitelik seti; TreeExplainer ile kesin ve hizli SHAP).
  1) Her egitim senaryosu (uc veri seti, yalnizca STEAD/INSTANCE/CEED) icin modelin KENDI
     bolge ici test setinde ve Turkiye'de (S varisi olan izler) ortalama |SHAP|
  2) Onem siralamalarinin bolgeler arasi Spearman korelasyonu
  3) Oznitelik gruplarinin (erken P, tam pencere, spektral, S-P) toplam katkisi
  4) Ana model icin Turkiye beeswarm grafigi

Gerekli: pip install shap
Cikti: DataBase/results/paper/ (Fig_shap_importance, Fig_shap_beeswarm, shap_*.csv)
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy import stats  # noqa: E402

from make_results import DATASETS, KOERI, MAIN, MM, OKABE, OUT, ROOT, SEED, SINGLE, save  # noqa: E402

N_SAMPLE = 5000
GROUPS = {"Early P (1–3 s)": ("Pd_", "Pv_", "Pa_", "PvVec_", "IV2_", "tauc_"),
          "Full window": ("log_PGV", "log_PGD", "log_PGA", "log_PGV_h", "log_CAV", "log_arias",
                          "log_energy_v", "dur_5_95_s"),
          "Spectral": ("dom_freq", "spec_centroid", "spec_entropy", "log_ratio_"),
          "S–P time": ("sp_time_s",)}
PRETTY = {"log_PGV": "log PGV", "log_PGD": "log PGD", "log_PGA": "log PGA", "log_PGV_h": "log PGV (horiz.)",
          "log_CAV": "log CAV", "log_arias": "log Arias", "log_energy_v": "log ∫v²", "dur_5_95_s": "D5–95",
          "dom_freq": "Dominant freq.", "spec_centroid": "Spectral centroid", "spec_entropy": "Spectral entropy",
          "log_ratio_low_mid": "log E(1–3)/E(3–8)", "log_ratio_mid_high": "log E(3–8)/E(8–20)",
          "sp_time_s": "S–P time"}


def pretty(f):
    if f in PRETTY:
        return PRETTY[f]
    for k in ("Pd", "Pv", "Pa", "PvVec", "IV2", "tauc"):
        if f.startswith(k + "_"):
            name = {"PvVec": "Pv (vector)", "IV2": "∫v²", "tauc": "τc"}.get(k, k)
            return f"{name} {f.split('_')[-1]}"
    return f


def group_of(f):
    for g, keys in GROUPS.items():
        if any(f.startswith(k) for k in keys):
            return g
    return "Other"


def mean_abs_shap(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    return np.abs(sv).mean(0), sv


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "features", "features_all.csv.gz"), low_memory=False)
    df = df[(df["split"] == "test") & (df["snr_db"] >= 20)].replace([np.inf, -np.inf], np.nan)
    tr = pd.read_csv(os.path.join(KOERI, "turkey_features.csv.gz"), dtype={"event_id": str})
    tr = tr[(tr["snr_db"] >= 20) & (~tr["clipped"].astype(bool)) & tr["has_S"].astype(bool)]
    tr = tr.replace([np.inf, -np.inf], np.nan)

    runs = {"All three": (MAIN, DATASETS)} | {d: (SINGLE[d], [d]) for d in DATASETS}
    imp, sv_main = {}, None
    for label, (run, src) in runs.items():
        path = os.path.join(ROOT, "results", run, "models.joblib")
        if not os.path.exists(path):
            continue
        b = joblib.load(path)
        feats, model = b["features"], b["models"]["LGBM"]
        dom = df[df["dataset"].isin(src)].sample(min(N_SAMPLE, (df["dataset"].isin(src)).sum()), random_state=SEED)
        tur = tr.sample(min(N_SAMPLE, len(tr)), random_state=SEED)
        imp[(label, "in-region")], _ = mean_abs_shap(model, dom[feats].values)
        imp[(label, "Türkiye")], sv = mean_abs_shap(model, tur[feats].values)
        if label == "All three":
            sv_main, X_main = sv, tur[feats]
        print(f"  {label}: SHAP tamam")

    I = pd.DataFrame(imp, index=feats)
    I.to_csv(os.path.join(OUT, "shap_importance.csv"))

    # Onem siralamalarinin tutarliligi (Spearman)
    cols = list(I.columns)
    rho = pd.DataFrame(index=[f"{a}|{b}" for a, b in cols], columns=[f"{a}|{b}" for a, b in cols], dtype=float)
    for c1 in cols:
        for c2 in cols:
            rho.loc[f"{c1[0]}|{c1[1]}", f"{c2[0]}|{c2[1]}"] = stats.spearmanr(I[c1], I[c2]).correlation
    rho.to_csv(os.path.join(OUT, "shap_rank_correlation.csv"))
    print("Spearman (onem siralamasi):")
    print(rho.round(2).to_string())

    # Grup katkilari
    G = I.groupby([group_of(f) for f in I.index]).sum()
    G = G / G.sum()
    G.to_csv(os.path.join(OUT, "shap_groups.csv"))

    # Sekil: ilk 12 oznitelik (ana model, bolge ici vs Turkiye) + grup paylari
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 75 * MM), gridspec_kw={"width_ratios": [1.4, 1]})
    if ("All three", "in-region") in I:
        top = I[("All three", "in-region")].sort_values(ascending=False).index[:12]
        y = np.arange(len(top))
        axes[0].barh(y - 0.2, I.loc[top, ("All three", "in-region")], 0.4, color=OKABE[5], label="In-region test")
        axes[0].barh(y + 0.2, I.loc[top, ("All three", "Türkiye")], 0.4, color=OKABE[1], label="Türkiye")
        axes[0].set_yticks(y, [pretty(f) for f in top])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Mean |SHAP| (magnitude units)")
        axes[0].legend(frameon=False)
        axes[0].set_title("LightGBM trained on all three benchmarks")
    groups = [g for g in GROUPS if g in G.index]
    x = np.arange(len(G.columns))
    bottom = np.zeros(len(G.columns))
    for g, c in zip(groups, [OKABE[5], OKABE[3], OKABE[1], OKABE[6]]):
        axes[1].bar(x, G.loc[g].values, bottom=bottom, color=c, label=g, width=0.7)
        bottom += G.loc[g].values
    axes[1].set_xticks(x, [f"{a}\n{b}" for a, b in G.columns], rotation=90, fontsize=6)
    axes[1].set_ylabel("Share of total |SHAP|")
    axes[1].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2)
    save(fig, "Fig_shap_importance")

    # Sekil: beeswarm (ana model, Turkiye)
    if sv_main is not None:
        plt.figure(figsize=(140 * MM, 90 * MM))
        shap.summary_plot(sv_main, X_main.rename(columns=pretty), max_display=12, show=False,
                          plot_size=None, color_bar_label="Feature value")
        save(plt.gcf(), "Fig_shap_beeswarm")
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()