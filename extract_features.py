"""
Oznitelik cikarimi: counts -> m/s (kazanc ile), fiziksel oznitelikler.
trace_index.csv.gz icindeki tum izler icin calisir; kaldigi yerden devam eder.

Kullanim:
    Linux:   nohup python extract_features.py --root ~/work/deprem/DataBase --workers 8 > features.log 2>&1 &
    Windows: python extract_features.py --root D:/deprem/DataBase --workers 8 > features.log 2>&1
             (Anaconda Prompt'tan calistirin; Spyder konsolundan degil)
Cikti:
    DataBase/features/<DATASET>_<parca>.csv.gz   (ara dosyalar)
    DataBase/features/features_all.csv.gz        (birlesik tablo)
"""
import os

# Windows: SeisBench'in yukledigi PyTorch (libomp) ile numpy/scipy'nin Intel MKL'si (libiomp5md)
# iki ayri OpenMP kutuphanesi getiriyor; ikisi ayni surece yuklenince Python cokuyor
# ("OMP: Error #15"). Oznitelik hesabinda cok is parcacigina gerek olmadigi icin tek is
# parcacigiyla calisip cakismaya izin veriyoruz. Bu ayarlar numpy'den ONCE yapilmali.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, sosfiltfilt, welch
from scipy.signal.windows import tukey

from common import cfg, load_sb_dataset

FS = 100.0
BAND = (1.0, 20.0)                  # Hz, tum veri setlerinde ortak bant
SOS = butter(4, BAND, btype="bandpass", fs=FS, output="sos")
PRE_S, POST_S = 5.0, 35.0           # P etrafinda kesilen parca (s)
P_WINDOWS = (1, 2, 3)               # P sonrasi erken pencereler (s)
FULL_WIN_S = 30.0                   # P sonrasi tam pencere (s)
SPEC_BANDS = [(1, 3), (3, 8), (8, 20)]
ID_COLS = ["dataset", "sb_index", "trace_name", "event_id", "split", "magnitude",
           "distance_km", "depth_km", "network", "station"]

DS = None  # SeisBench veri seti (Linux: fork ile aktarilir; Windows: her surecte acilir)


def _init_worker(root, name):
    """Windows (spawn) icin: her isci surec veri setini kendisi acar."""
    global DS
    DS = load_sb_dataset(name, root)


def log10p(x):
    return float(np.log10(x)) if x > 0 else np.nan


def linear_detrend(x):
    """Dogrusal egilimi kaldirir (scipy.signal.detrend(type='linear') ile ayni sonuc).
    Kapali form; LAPACK/MKL cagirmadigi icin OpenMP cakismasina takilmaz."""
    t = np.arange(x.shape[-1], dtype=np.float64)
    t -= t.mean()
    xm = x.mean(axis=-1, keepdims=True)
    slope = ((x - xm) * t).sum(axis=-1, keepdims=True) / (t * t).sum()
    return x - xm - slope * t


def ref_snr(raw, p):
    """Celik (2026) ile ayni SNR tanimi (ham counts): karsilastirma icin."""
    if p < 60 or p + 200 > raw.shape[1]:
        return np.nan
    vals = []
    for ch in range(3):
        noise = max(np.mean(raw[ch, p - 60:p - 10] ** 2), 1e-2)
        sig = np.mean(raw[ch, p:p + 200] ** 2)
        vals.append(10 * np.log10(sig / noise) if sig > 0 else np.nan)
    return float(np.nanmean(vals))


def compute_features(raw, sens, p, s):
    """raw: (3, N) counts, ZNE sirasi; sens: (3,) count/(m/s); p, s: ornek indeksi."""
    n = raw.shape[1]
    if not np.isfinite(p) or p < 2 * FS or p + FS > n:
        return None
    p = int(round(p))
    f = {"snr_ref_db": ref_snr(raw.astype(np.float64), p)}

    # P etrafinda kes, m/s'ye cevir, on isle
    a0, a1 = max(0, p - int(PRE_S * FS)), min(n, p + int(POST_S * FS))
    v = raw[:, a0:a1].astype(np.float64) / sens[:, None]
    v = linear_detrend(v)
    v *= tukey(v.shape[1], alpha=0.05)
    v = sosfiltfilt(SOS, v, axis=1)
    d = sosfiltfilt(SOS, cumulative_trapezoid(v, dx=1 / FS, initial=0, axis=1), axis=1)
    acc = np.gradient(v, 1 / FS, axis=1)
    pp = p - a0                       # kesilmis parcada P indeksi
    m = v.shape[1]

    # Gurultu seviyesi ve kendi SNR tanimimiz
    noise = v[:, max(0, pp - int(1.5 * FS)):pp - int(0.5 * FS)]
    vvec = np.sqrt((v ** 2).sum(axis=0))
    noise_rms = np.sqrt(np.mean((noise ** 2).sum(axis=0))) if noise.size else np.nan
    f["log_noise_rms"] = log10p(noise_rms)

    # Erken P penceresi oznitelikleri (dusey bilesen agirlikli)
    for w in P_WINDOWS:
        e = pp + int(w * FS)
        if e > m:
            for k in ("Pd", "Pv", "Pa", "PvVec", "IV2", "tauc"):
                f[f"{k}_{w}s"] = np.nan
            continue
        vz, dz, az = v[0, pp:e], d[0, pp:e], acc[0, pp:e]
        f[f"Pd_{w}s"] = log10p(np.max(np.abs(dz)))
        f[f"Pv_{w}s"] = log10p(np.max(np.abs(vz)))
        f[f"Pa_{w}s"] = log10p(np.max(np.abs(az)))
        f[f"PvVec_{w}s"] = log10p(np.max(vvec[pp:e]))
        f[f"IV2_{w}s"] = log10p(np.sum(vz ** 2) / FS)
        den = np.sum(vz ** 2)
        f[f"tauc_{w}s"] = float(2 * np.pi * np.sqrt(np.sum(dz ** 2) / den)) if den > 0 else np.nan

    # Tam pencere (P'den itibaren en fazla 30 s)
    e = min(m, pp + int(FULL_WIN_S * FS))
    vw, dw, aw = v[:, pp:e], d[:, pp:e], acc[:, pp:e]
    avec = np.sqrt((aw ** 2).sum(axis=0))
    dvec = np.sqrt((dw ** 2).sum(axis=0))
    vvw = vvec[pp:e]
    f["log_PGV"] = log10p(vvw.max())
    f["log_PGD"] = log10p(dvec.max())
    f["log_PGA"] = log10p(avec.max())
    f["log_PGV_h"] = log10p(np.sqrt(vw[1] ** 2 + vw[2] ** 2).max())
    f["log_CAV"] = log10p(np.sum(avec) / FS)
    f["log_arias"] = log10p(np.sum(avec ** 2) / FS)
    f["log_energy_v"] = log10p(np.sum(vvw ** 2) / FS)
    f["snr_db"] = (20 * np.log10(vvw.max() / noise_rms)
                   if noise_rms and np.isfinite(noise_rms) and noise_rms > 0 else np.nan)
    cum = np.cumsum(vvw ** 2)
    if cum[-1] > 0:
        i5, i95 = np.searchsorted(cum, 0.05 * cum[-1]), np.searchsorted(cum, 0.95 * cum[-1])
        f["dur_5_95_s"] = (i95 - i5) / FS
    else:
        f["dur_5_95_s"] = np.nan

    # Spektral oznitelikler (dusey hiz, tam pencere)
    fr, psd = welch(vw[0], fs=FS, nperseg=min(256, vw.shape[1]))
    sel = (fr >= BAND[0]) & (fr <= BAND[1])
    fr, psd = fr[sel], psd[sel]
    tot = psd.sum()
    if tot > 0:
        pn = psd / tot
        f["dom_freq"] = float(fr[np.argmax(psd)])
        f["spec_centroid"] = float(np.sum(fr * pn))
        f["spec_entropy"] = float(-np.sum(pn * np.log(pn + 1e-12)) / np.log(len(pn)))
        be = [psd[(fr >= lo) & (fr < hi)].sum() for lo, hi in SPEC_BANDS]
        f["log_ratio_low_mid"] = log10p(be[0] / be[1]) if be[1] > 0 else np.nan
        f["log_ratio_mid_high"] = log10p(be[1] / be[2]) if be[2] > 0 else np.nan
    else:
        for k in ("dom_freq", "spec_centroid", "spec_entropy", "log_ratio_low_mid", "log_ratio_mid_high"):
            f[k] = np.nan

    # S-P suresi (yoksa NaN; gradient boosting eksik degeri dogrudan isler)
    f["sp_time_s"] = (s - p) / FS if np.isfinite(s) and s > p else np.nan
    return f


def worker(row):
    sb_index, sz, sn, se, p, s = row
    try:
        raw = DS.get_waveforms(int(sb_index))
        feats = compute_features(raw, np.array([sz, sn, se]), p, s)
    except Exception:
        feats = None
    return sb_index, feats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=cfg("ROOT"), required=cfg("ROOT") is None)
    ap.add_argument("--workers", type=int, default=cfg("WORKERS", 8))
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--datasets", nargs="+", default=cfg("DATASETS", ["STEAD", "INSTANCE"]))
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.root)
    out_dir = os.path.join(root, "features")
    os.makedirs(out_dir, exist_ok=True)
    idx = pd.read_csv(os.path.join(root, "harmonized", "trace_index.csv.gz"), low_memory=False)

    global DS
    for name in args.datasets:
        sub = idx[idx["dataset"] == name].sort_values("sb_index").reset_index(drop=True)
        print(f"\n{name}: {len(sub):,} iz", flush=True)
        DS = load_sb_dataset(name, root)
        assert DS.component_order == "ZNE", DS.component_order
        check = sub.sample(min(20, len(sub)), random_state=0)
        assert (DS.metadata.loc[check["sb_index"], "trace_name"].values
                == check["trace_name"].values).all(), "Indeks uyusmazligi!"

        if args.workers <= 1:
            pool = None                      # tek surec: paralel isleme sorun cikarirsa guvenli yol
        elif sys.platform == "win32" or os.environ.get("FORCE_SPAWN"):
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(args.workers, initializer=_init_worker, initargs=(root, name))
        else:
            ctx = mp.get_context("fork")
            pool = ctx.Pool(args.workers)
        print(f"  {'tek surec' if pool is None else str(args.workers) + ' surec'} hazir; isleme basliyor",
              flush=True)
        for ci, start in enumerate(range(0, len(sub), args.chunk)):
            out = os.path.join(out_dir, f"{name}_{ci:04d}.csv.gz")
            part = sub.iloc[start:start + args.chunk]
            if os.path.exists(out):
                # Guvenlik: iz tablosu yeniden uretildiyse eski parca dosyasi uyusmayabilir
                prev = pd.read_csv(out, usecols=["sb_index"])["sb_index"]
                if prev.isin(part["sb_index"]).all() and len(prev) >= 0.95 * len(part):
                    continue
                print(f"  parca {ci}: eski dosya iz tablosuyla uyusmuyor, yeniden hesaplaniyor")
            rows = list(part[["sb_index", "sens_Z", "sens_N", "sens_E",
                              "p_sample", "s_sample"]].itertuples(index=False, name=None))
            t0 = time.time()
            res = pool.map(worker, rows, chunksize=100) if pool else [worker(r) for r in rows]
            feats = {k: v for k, v in res if v is not None}
            fdf = pd.DataFrame.from_dict(feats, orient="index")
            fdf.index.name = "sb_index"
            merged = part[ID_COLS].merge(fdf.reset_index(), on="sb_index", how="inner")
            merged.to_csv(out, index=False)
            print(f"  parca {ci}: {len(merged):,}/{len(part):,} basarili, "
                  f"{time.time() - t0:.0f} s", flush=True)
        if pool:
            pool.close()
            pool.join()

    files = sorted(glob.glob(os.path.join(out_dir, "*_[0-9][0-9][0-9][0-9].csv.gz")))
    allf = pd.concat((pd.read_csv(fp) for fp in files), ignore_index=True)
    allf.to_csv(os.path.join(out_dir, "features_all.csv.gz"), index=False)
    print(f"\nToplam: {len(allf):,} iz, {allf.shape[1]} sutun")
    print(allf.groupby(["dataset", "split"]).size().to_string())


if __name__ == "__main__":
    main()
    
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# -------------------------------------------------------------
# Dizin Tanımlaması:
# Diğer kodunuzda kullandığınız bir klasör yolu (output_dir) varsa 
# doğrudan buraya yazabilirsiniz. Boş bırakılırsa dosyanın çalıştığı
# mevcut çalışma dizinine (current working directory) kaydeder.
# -------------------------------------------------------------
output_dir = "."  # Örneğin: "figures" veya r"C:\...\figures"

os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, "Fig_diversity_fixed.pdf")
png_path = os.path.join(output_dir, "Fig_diversity_fixed.png")

# 1. Model İsimleri ve Ver    