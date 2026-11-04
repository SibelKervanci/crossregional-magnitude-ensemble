"""
Sizinti denetimi (egitimden ONCE bir kez calistirilir).

1) Veri setleri arasi mukerrer depremler: ayni deprem iki veri setinde birden
   varsa bolgeler arasi test sizdirir. TUM veri seti ciftleri kontrol edilir
   (STEAD-INSTANCE, STEAD-CEED, INSTANCE-CEED). Origin zamani (<= 5 s) ve
   episantr uzakligi (<= 50 km) ile eslestirilir. Mukerrer deprem, oncelik
   sirasinda once gelen veri setinden cikarilir: STEAD > CEED > INSTANCE
   (STEAD kuresel ve karisik oldugu icin; bolgesel veri setleri bozulmadan kalir).
2) Turkiye bolgesi: dis dogrulama bolgesindeki depremler ve istasyonlar egitim
   verisinden tamamen cikarilir.

Kullanim:
    Spyder: F5 (ayarlar config.py'den)
    Komut satiri: python audit_leakage.py --root ~/work/deprem/DataBase > audit_rapor.txt
Cikti:
    DataBase/harmonized/exclude_events.csv  (train_models.py otomatik okur)
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from common import cfg, load_sb_dataset

TIME_TOL_S = 5.0
DIST_TOL_KM = 50.0
TR_BOX = dict(lat_min=34.0, lat_max=43.5, lon_min=24.0, lon_max=46.0)
REMOVE_PRIORITY = ["STEAD", "CEED", "INSTANCE"]     # mukerrer: listede once gelenden cikar
CSV_META = {"STEAD": "stead", "INSTANCE": "instancecounts"}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def in_tr_box(lat, lon):
    return ((lat >= TR_BOX["lat_min"]) & (lat <= TR_BOX["lat_max"])
            & (lon >= TR_BOX["lon_min"]) & (lon <= TR_BOX["lon_max"]))


def load_events(root, name, idx):
    """Veri setindeki (iz tablosunda kullanilan) depremlerin zaman ve konumlari."""
    sub = idx[idx["dataset"] == name]
    if name in CSV_META:
        cols = ["source_id", "source_origin_time", "source_latitude_deg", "source_longitude_deg",
                "station_latitude_deg", "station_longitude_deg"]
        md = pd.read_csv(os.path.join(root, "datasets", CSV_META[name], "metadata.csv"),
                         usecols=cols, low_memory=False)
        md["event_id"] = f"{name}_" + md["source_id"].astype(str)
        md["sta_in_tr"] = in_tr_box(md["station_latitude_deg"], md["station_longitude_deg"])
    else:  # CEED: SeisBench indeksleriyle
        md = load_sb_dataset("CEED", root).metadata.loc[sub["sb_index"]].copy()
        md["event_id"] = sub["event_id"].values
        md["sta_in_tr"] = False   # CEED istasyonlari Kaliforniya'da
    md = md[md["event_id"].isin(set(sub["event_id"]))]
    md["origin"] = pd.to_datetime(md["source_origin_time"], utc=True, format="mixed", errors="coerce")
    ev = md.groupby("event_id").agg(
        origin=("origin", "first"), lat=("source_latitude_deg", "first"),
        lon=("source_longitude_deg", "first"), any_sta_in_tr=("sta_in_tr", "any")).reset_index()
    ev["dataset"] = name
    print(f"  {name}: {len(ev):,} deprem", flush=True)
    return ev


def find_duplicates(a, b):
    a = a.dropna(subset=["origin"]).sort_values("origin")
    b = b.dropna(subset=["origin"]).sort_values("origin")
    m = pd.merge_asof(a, b, on="origin", direction="nearest",
                      tolerance=pd.Timedelta(seconds=TIME_TOL_S), suffixes=("", "_b"))
    m = m.dropna(subset=["event_id_b"])
    m["dist_km"] = haversine_km(m["lat"], m["lon"], m["lat_b"], m["lon_b"])
    return m[m["dist_km"] <= DIST_TOL_KM]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=cfg("ROOT"), required=cfg("ROOT") is None)
    args = ap.parse_args(argv)
    root = os.path.expanduser(args.root)

    idx = pd.read_csv(os.path.join(root, "harmonized", "trace_index.csv.gz"),
                      usecols=["dataset", "sb_index", "event_id", "network"], low_memory=False)
    datasets = [d for d in REMOVE_PRIORITY if d in idx["dataset"].unique()]
    print("Depremler yukleniyor...")
    events = {d: load_events(root, d, idx) for d in datasets}
    exclude = []

    print("\n=== 1) Veri setleri arasi mukerrer depremler ===")
    for d1, d2 in itertools.combinations(datasets, 2):   # d1 oncelikli (cikarilacak taraf)
        dup = find_duplicates(events[d1], events[d2])
        print(f"{d1}-{d2} mukerrer deprem: {len(dup):,} ({d1} tarafindan cikarilacak)")
        exclude += [(e, "duplicate_of_" + o) for e, o in zip(dup["event_id"], dup["event_id_b"])]

    print("\n=== 2) Turkiye bolgesi (dis dogrulama bolgesi) ===")
    allev = pd.concat(events.values(), ignore_index=True)
    ev_tr = allev[in_tr_box(allev["lat"], allev["lon"])]
    sta_tr = allev[allev["any_sta_in_tr"]]
    print(f"Episantri Turkiye kutusunda olan deprem: {len(ev_tr):,}")
    print(f"Turkiye kutusundaki bir istasyonda kaydedilmis deprem: {len(sta_tr):,}")
    for net in ["KO", "TU", "TK", "TL"]:
        n = (idx["network"] == net).sum()
        print(f"  {net} agi iz sayisi: {n}" + ("  <-- cikarilacak" if n else ""))
    exclude += [(e, "epicenter_in_turkiye_box") for e in ev_tr["event_id"]]
    exclude += [(e, "station_in_turkiye_box") for e in sta_tr["event_id"]]

    ex = pd.DataFrame(exclude, columns=["event_id", "reason"]).drop_duplicates("event_id")
    out = os.path.join(root, "harmonized", "exclude_events.csv")
    ex.to_csv(out, index=False)
    n_tr = idx["event_id"].isin(ex["event_id"]).sum()
    print(f"\nCikarilacak deprem: {len(ex):,} ({n_tr:,} iz, %{100 * n_tr / len(idx):.2f}). Kaydedildi: {out}")
    print(ex.assign(dataset=ex["event_id"].str.split("_").str[0],
                    reason=ex["reason"].str.replace(r"duplicate_of_.*", "duplicate", regex=True))
          .groupby(["dataset", "reason"]).size().to_string())


if __name__ == "__main__":
    main()
