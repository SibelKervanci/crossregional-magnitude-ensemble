"""
Birlesik iz tablosu (trace index): filtreleme + kazanc eslestirme + olay bazli bolme.
Waveform dosyalarina dokunmaz; yalnizca metadata ve station_gains.csv kullanir.

Kullanim:
    Spyder: F5 (ayarlar config.py'den)
    Komut satiri: python build_trace_index.py --root ~/work/deprem/DataBase > trace_index_rapor.txt
CEED: veriler fiziksel birimde (1e-6 m/s) geldigi icin kazanc eslestirmesi gerekmez;
      kazanc 1e6 alinarak ayni donusum (deger / kazanc = m/s) uygulanir.
Cikti:
    DataBase/harmonized/trace_index.csv.gz
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import cfg, load_sb_dataset

VEL_BANDS = ["HH", "BH", "EH"]
COMP_MAP = {"Z": "Z", "N": "N", "1": "N", "E": "E", "2": "E"}
MAX_DIST_KM = 110
SENS_MIN, SENS_MAX = 1e6, 1e12          # makul kazanc araligi, count/(m/s)
AMBIGUITY_RATIO = 1.05                  # ayni anda farkli kazancli sensorler
FAR_FUTURE = pd.Timestamp("2100-01-01", tz="UTC")
MAG_BINS = [0, 1, 2, 3, 4, 5, 10]
CHUNK = 200_000


def to_utc(series):
    return pd.to_datetime(series, utc=True, format="mixed", errors="coerce")


def load_stead(base):
    cols = ["trace_name", "source_id", "source_magnitude", "source_magnitude_type",
            "source_distance_km", "source_depth_km", "station_network_code", "station_code",
            "trace_channel", "trace_start_time", "trace_p_arrival_sample",
            "trace_s_arrival_sample", "trace_snr_db"]
    md = pd.read_csv(os.path.join(base, "stead", "metadata.csv"), usecols=cols, low_memory=False)
    md = md[(md["source_magnitude_type"].astype(str).str.lower() == "ml")
            & (md["source_magnitude"] >= 0)
            & (md["source_distance_km"] <= MAX_DIST_KM)
            & (md["trace_channel"].isin(VEL_BANDS))]
    snr = (md["trace_snr_db"].astype(str).str.strip("[]").str.split(expand=True)
           .iloc[:, :3].apply(pd.to_numeric, errors="coerce"))
    return pd.DataFrame({
        "dataset": "STEAD", "sb_index": md.index, "trace_name": md["trace_name"],
        "event_id": "STEAD_" + md["source_id"].astype(str),
        "magnitude": md["source_magnitude"], "distance_km": md["source_distance_km"],
        "depth_km": md["source_depth_km"], "network": md["station_network_code"],
        "station": md["station_code"], "band": md["trace_channel"],
        "start_time": to_utc(md["trace_start_time"]),
        "p_sample": md["trace_p_arrival_sample"], "s_sample": md["trace_s_arrival_sample"],
        "snr_mean": snr.mean(axis=1).values, "snr_min": snr.min(axis=1).values,
        "vs30": np.nan,
    })


def load_instance(base):
    cols = ["trace_name", "source_id", "source_magnitude", "source_magnitude_type",
            "path_ep_distance_km", "source_depth_km", "station_network_code", "station_code",
            "station_channels", "trace_start_time", "trace_P_arrival_sample",
            "trace_S_arrival_sample", "trace_E_snr_db", "trace_N_snr_db", "trace_Z_snr_db",
            "station_vs_30_mps"]
    md = pd.read_csv(os.path.join(base, "instancecounts", "metadata.csv"), usecols=cols, low_memory=False)
    md = md[(md["source_magnitude_type"].astype(str).str.lower() == "ml")
            & (md["source_magnitude"] >= 0)
            & (md["path_ep_distance_km"] <= MAX_DIST_KM)
            & (md["station_channels"].isin(VEL_BANDS))]
    snr = md[["trace_E_snr_db", "trace_N_snr_db", "trace_Z_snr_db"]]
    return pd.DataFrame({
        "dataset": "INSTANCE", "sb_index": md.index, "trace_name": md["trace_name"],
        "event_id": "INSTANCE_" + md["source_id"].astype(str),
        "magnitude": md["source_magnitude"], "distance_km": md["path_ep_distance_km"],
        "depth_km": md["source_depth_km"], "network": md["station_network_code"],
        "station": md["station_code"], "band": md["station_channels"],
        "start_time": to_utc(md["trace_start_time"]),
        "p_sample": md["trace_P_arrival_sample"], "s_sample": md["trace_S_arrival_sample"],
        "snr_mean": snr.mean(axis=1).values, "snr_min": snr.min(axis=1).values,
        "vs30": md["station_vs_30_mps"],
    })


def parse_snr(series):
    """SNR sutunu tek sayi ya da '[a b c]' biciminde olabilir; (ortalama, en kucuk) dondurur."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.5:
        return num.values, num.values
    arr = (series.astype(str).str.strip("[]").str.replace(",", " ")
           .str.split(expand=True).apply(pd.to_numeric, errors="coerce"))
    return arr.mean(axis=1).values, arr.min(axis=1).values


def load_ceed(root):
    md = load_sb_dataset("CEED", root).metadata
    mt = md["source_magnitude_type"].astype(str).str.strip().str.lower()
    md = md[(mt == "l")
            & (md["source_magnitude"] >= 0)
            & (md["path_ep_distance_km"] <= MAX_DIST_KM)
            & (md["station_instrument"].isin(VEL_BANDS))
            & (md["trace_unit"].astype(str) == "1e-6m/s")]
    snr_mean, snr_min = parse_snr(md["trace_snr_db"])
    eid = md["source_id_list"].astype(str).str.extract(r"([A-Za-z]*\d+)")[0]
    return pd.DataFrame({
        "dataset": "CEED", "sb_index": md.index, "trace_name": md["trace_name"],
        "event_id": "CEED_" + eid, "magnitude": md["source_magnitude"],
        "distance_km": md["path_ep_distance_km"], "depth_km": md["source_depth_km"],
        "network": md["station_network_code"], "station": md["station_code"],
        "band": md["station_instrument"], "start_time": to_utc(md["trace_start_time"]),
        "p_sample": md["trace_p_arrival_sample"], "s_sample": md["trace_s_arrival_sample"],
        "snr_mean": snr_mean, "snr_min": snr_min, "vs30": np.nan,
        "sens_Z": 1e6, "sens_N": 1e6, "sens_E": 1e6,     # 1e-6 m/s -> m/s
    })


def load_gains(path):
    g = pd.read_csv(path, low_memory=False)
    g = g[g["input_units"].astype(str).str.lower() == "m/s"].copy()
    g["sensitivity"] = g["sensitivity"].abs()
    g = g[(g["sensitivity"] >= SENS_MIN) & (g["sensitivity"] <= SENS_MAX)]
    g["band"] = g["channel"].str[:2]
    g["comp"] = g["channel"].str[2].map(COMP_MAP)
    g = g.dropna(subset=["comp"])
    g["start"] = to_utc(g["start"])
    g["end"] = to_utc(g["end"]).fillna(FAR_FUTURE)
    return g[["network", "station", "band", "comp", "start", "end", "sensitivity"]]


def attach_gains(tr, gains):
    """Her ize Z/N/E kazanci atar; zaman donemine gore eslestirir."""
    tr = tr.reset_index(drop=True)
    parts = []
    for i in range(0, len(tr), CHUNK):
        chunk = tr.iloc[i:i + CHUNK][["network", "station", "band", "start_time"]].copy()
        chunk["tid"] = chunk.index
        m = chunk.merge(gains, on=["network", "station", "band"], how="inner")
        m = m[(m["start"] <= m["start_time"]) & (m["start_time"] < m["end"])]
        agg = m.groupby(["tid", "comp"])["sensitivity"].agg(["min", "max", "median"])
        agg = agg[agg["max"] / agg["min"] <= AMBIGUITY_RATIO]
        parts.append(agg["median"].unstack("comp"))
    piv = pd.concat(parts) if parts else pd.DataFrame()
    for c in "ZNE":
        tr[f"sens_{c}"] = piv[c].reindex(tr.index).values if c in piv else np.nan
    return tr


def assign_splits(df, seed):
    ev = df.groupby("event_id")["magnitude"].first().reset_index()
    ev["mbin"] = pd.cut(ev["magnitude"], MAG_BINS, include_lowest=True)
    ev["split"] = ""
    rng = np.random.default_rng(seed)
    for _, grp in ev.groupby("mbin", observed=True):
        idx = rng.permutation(grp.index.values)
        n = len(idx)
        ev.loc[idx[:int(0.70 * n)], "split"] = "train"
        ev.loc[idx[int(0.70 * n):int(0.85 * n)], "split"] = "dev"
        ev.loc[idx[int(0.85 * n):], "split"] = "test"
    return df.merge(ev[["event_id", "split"]], on="event_id", how="left")


def process(name, tr, gains, args, log):
    log.append((name, "1) ML, <=110 km, hiz sensoru", len(tr), tr["event_id"].nunique()))
    if "sens_Z" not in tr.columns:                     # CEED zaten fiziksel birimde
        tr = attach_gains(tr, gains)
    tr = tr.dropna(subset=["sens_Z", "sens_N", "sens_E", "event_id"])
    log.append((name, "2) Uc bilesen kazanci eslesti", len(tr), tr["event_id"].nunique()))
    if args.min_snr is not None:
        tr = tr[tr["snr_mean"] >= args.min_snr]
        log.append((name, f"3) Veri seti SNR >= {args.min_snr} dB", len(tr), tr["event_id"].nunique()))
    tr = tr.sort_values("distance_km").groupby("event_id").head(args.max_per_event)
    log.append((name, f"3) Olay basina en fazla {args.max_per_event} iz (en yakin)", len(tr),
                tr["event_id"].nunique()))
    return assign_splits(tr, args.seed)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=cfg("ROOT"), required=cfg("ROOT") is None)
    p.add_argument("--datasets", nargs="+", default=cfg("DATASETS", ["STEAD", "INSTANCE"]))
    p.add_argument("--min-snr", type=float, default=None,
                   help="Veri setinin KENDI SNR sutunuyla filtre (varsayilan: kapali). SNR tanimlari "
                        "veri setleri arasinda farkli oldugu icin kalite filtresi oznitelik "
                        "cikarimindan sonra ortak snr_db ile uygulanir (train_models.py --min-snr).")
    p.add_argument("--max-per-event", type=int, default=5)
    p.add_argument("--seed", type=int, default=cfg("SEED", 42))
    args = p.parse_args(argv)

    root = os.path.expanduser(args.root)
    base = os.path.join(root, "datasets")
    gains = load_gains(os.path.join(root, "station_gains.csv"))
    print(f"Gecerli kazanc donemi: {len(gains):,}")

    loaders = {"STEAD": lambda: load_stead(base), "INSTANCE": lambda: load_instance(base),
               "CEED": lambda: load_ceed(root)}
    log, frames = [], []
    for name in args.datasets:
        print(f"\n{name} isleniyor...", flush=True)
        frames.append(process(name, loaders[name](), gains, args, log))
    df = pd.concat(frames, ignore_index=True)

    out_dir = os.path.join(root, "harmonized")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "trace_index.csv.gz")
    df.to_csv(out, index=False)

    print("\n=== Asama asama iz ve olay sayilari ===")
    print(pd.DataFrame(log, columns=["Veri seti", "Asama", "Iz", "Olay"]).to_string(index=False))

    print("\n=== Bolmeler (iz / olay) ===")
    print(df.groupby(["dataset", "split"]).agg(iz=("trace_name", "size"),
                                                olay=("event_id", "nunique")).to_string())

    print("\n=== Buyukluk araliklari (iz sayisi) ===")
    df["mbin"] = pd.cut(df["magnitude"], MAG_BINS, include_lowest=True)
    print(pd.crosstab([df["dataset"], df["split"]], df["mbin"]).to_string())

    leak = df.groupby("event_id")["split"].nunique().gt(1).sum()
    print(f"\nBirden fazla bolmede yer alan olay: {leak} (0 olmali)")
    print(f"\nKaydedildi: {out} ({len(df):,} iz)")


if __name__ == "__main__":
    main()