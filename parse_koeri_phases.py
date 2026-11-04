"""
KOERI yillik faz okuma dosyalarini (GSE2.0 bulten formati) okur.

Uretilen dosyalar (--out klasorune):
    koeri_events.csv : deprem basliklari (kod, olus zamani, konum, derinlik, ML, kalite)
    koeri_picks.csv  : faz okumalari (kod, istasyon, faz, varis zamani, uzaklik, istasyon ML)
    koeri_first_arrivals.csv : her deprem-istasyon cifti icin ilk P ve ilk S varisi

Istege bagli olarak KOERI katalog sorgusu (.txt) ile eslestirir ve ML farklarini raporlar.

Kullanim:
    python parse_koeri_phases.py --faz D:/sibel/deprem/KOERI/faz --out D:/sibel/deprem/KOERI ^
        --catalog D:/sibel/deprem/KOERI/20230206_20261026_2_5_9_0_23_948.txt --years 2023 2024 2025 2026
"""
import argparse
import glob
import os
import re

import pandas as pd
from common import cfg


RE_EVENT = re.compile(r"^EVENT\s+(\d{14})")
RE_ORIGIN = re.compile(
    r"^(\d{4}/\d\d/\d\d)\s+(\d\d:\d\d:\d\d(?:\.\d+)?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.?\d*)"
    r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(-?\d+\.\d+)")
RE_PHASE = re.compile(
    r"^(\S+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+([am])\s+(\S+)\s+(\d{4}/\d\d/\d\d)\s+(\d\d:\d\d:\d\d(?:\.\d+)?)"
    r"\s+(-?\d+\.\d+)?")
RE_AMP_MAG = re.compile(r"\bT\s+(-?\d+\.\d+)\s+(?:(-?\d+\.\d+)\s+)?ML\s+(-?\d+\.\d+)")
RE_QUALITY = re.compile(r"\b([amg])\s+([ifg])\s+(\w\w)\s*$")


def parse_file(path):
    events, picks = [], []
    cur = None
    expect_quality = False
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = RE_EVENT.match(line)
            if m:
                cur = {"event_id": m.group(1)}
                events.append(cur)
                expect_quality = False
                continue
            if cur is None:
                continue
            m = RE_ORIGIN.match(line)
            if m and "origin_time" not in cur:
                cur.update(origin_time=f"{m.group(1).replace('/', '-')}T{m.group(2)}",
                           lat=float(m.group(3)), lon=float(m.group(4)), depth_km=float(m.group(5)),
                           ndef=int(m.group(6)), nsta=int(m.group(7)), gap=int(m.group(8)),
                           mag_type=m.group(9), mag=float(m.group(10)))
                expect_quality = True
                continue
            if expect_quality:
                q = RE_QUALITY.search(line)
                if q:
                    cur.update(analysis=q.group(1), loc_method=q.group(2), event_type=q.group(3))
                expect_quality = False
                continue
            m = RE_PHASE.match(line)
            if m and not line.startswith("Sta "):
                am = RE_AMP_MAG.search(line)
                picks.append({
                    "event_id": cur["event_id"], "station": m.group(1),
                    "dist_deg": float(m.group(2)), "evaz": float(m.group(3)),
                    "pick_mode": m.group(4), "phase": m.group(5),
                    "time": f"{m.group(6).replace('/', '-')}T{m.group(7)}",
                    "tres": float(m.group(8)) if m.group(8) else None,
                    "amp": float(am.group(1)) if am else None,
                    "sta_ml": float(am.group(3)) if am else None,
                })
    return events, picks


def first_arrivals(picks):
    p = picks.copy()
    p["t"] = pd.to_datetime(p["time"], utc=True, format="mixed")
    p["kind"] = p["phase"].str[0].str.upper()
    p = p[p["kind"].isin(["P", "S"])]
    first = (p.sort_values("t").groupby(["event_id", "station", "kind"], as_index=False)
             .agg(time=("t", "first"), phase=("phase", "first"), dist_deg=("dist_deg", "first"),
                  sta_ml=("sta_ml", "max"), mode=("pick_mode", "first")))
    wide = first.pivot_table(index=["event_id", "station"], columns="kind",
                             values=["time", "phase", "mode"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    meta = first.groupby(["event_id", "station"], as_index=False).agg(
        dist_deg=("dist_deg", "first"), sta_ml=("sta_ml", "max"))
    out = meta.merge(wide, on=["event_id", "station"])
    out["dist_km"] = out["dist_deg"] * 111.19
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default=cfg("KOERI_FAZ"), required=cfg("KOERI_FAZ") is None,
                    help="Yillik .txt faz dosyalarinin klasoru")
    ap.add_argument("--out", default=cfg("KOERI"), required=cfg("KOERI") is None)
    ap.add_argument("--years", nargs="*", type=int, default=cfg("KOERI_YEARS"), help="Bos = tum yillar")
    ap.add_argument("--catalog", default=cfg("KOERI_CATALOG"),
                    help="KOERI katalog sorgu dosyasi (.txt); verilirse yalnizca bu depremler")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.faz, "*.txt")))
    if args.years:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in map(str, args.years)]
    all_ev, all_pk = [], []
    for f in files:
        ev, pk = parse_file(f)
        print(f"{os.path.basename(f)}: {len(ev):,} deprem, {len(pk):,} okuma", flush=True)
        all_ev += ev
        all_pk += pk
    events = pd.DataFrame(all_ev)
    picks = pd.DataFrame(all_pk)
    n_dup = events["event_id"].duplicated().sum()
    if n_dup:
        print(f"Bultende tekrarlanan deprem kodu: {n_dup} (ilk kayit tutuldu)")
        events = events.drop_duplicates("event_id", keep="first")
    picks = picks.drop_duplicates(["event_id", "station", "phase", "time"])

    if args.catalog:
        cat = pd.read_csv(args.catalog, sep="\t", encoding="latin-1")
        cat.columns = [c.strip() for c in cat.columns]
        cat["event_id"] = cat["Deprem Kodu"].astype(str).str.strip()
        cat = cat.drop_duplicates("event_id")
        before = len(events)
        events = events.merge(cat[["event_id", "ML", "xM"]].rename(columns={"ML": "catalog_ML"}),
                              on="event_id", how="inner")
        picks = picks[picks["event_id"].isin(events["event_id"])]
        missing = len(set(cat["event_id"]) - set(events["event_id"]))
        print(f"\nKatalog eslesmesi: {len(events):,}/{cat['event_id'].nunique():,} katalog depremi bultende "
              f"bulundu ({missing:,} eksik); bultende toplam {before:,} deprem vardi.")
        d = (events["mag"] - events["catalog_ML"]).dropna()
        print(f"Bulten ML - katalog ML: ortalama {d.mean():+.3f}, |fark|>0.05 olan {(d.abs() > 0.05).mean():.1%}")

    fa = first_arrivals(picks)
    events.to_csv(os.path.join(args.out, "koeri_events.csv"), index=False)
    picks.to_csv(os.path.join(args.out, "koeri_picks.csv"), index=False)
    fa.to_csv(os.path.join(args.out, "koeri_first_arrivals.csv"), index=False)

    print(f"\nDeprem: {len(events):,} | okuma: {len(picks):,} | deprem-istasyon cifti: {len(fa):,}")
    print("Okuma turu (pick_mode):", picks["pick_mode"].value_counts().to_dict())
    print("Analiz turu (analysis):", events.get("analysis", pd.Series(dtype=str)).value_counts().to_dict())
    print("Faz turleri:", picks["phase"].value_counts().head(8).to_dict())
    near = fa[fa["dist_km"] <= 110]
    print(f"<=110 km cift: {len(near):,} | S varisi olan: {near['time_S'].notna().mean():.1%}")
    print("En cok okumasi olan istasyonlar (<=110 km):")
    print(near["station"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
