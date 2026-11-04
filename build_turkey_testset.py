"""
Turkiye (Guneydogu, 2023-2026) dis test seti.

Adimlar:
  1) Istasyon envanteri: KO agindaki hiz sensorlu istasyonlarin StationXML'leri (kazanc).
  2) Secim: <=110 km, P okumasi olan deprem-istasyon ciftleri; deprem basina en yakin
     5 istasyon (egitim verisiyle ayni kural).
  3) Indirme: her deprem icin KOERI FDSN'den P-20 s ile P+40 s arasi kayitlar
     (MiniSEED, deprem basina bir dosya). Kaldigi yerden devam eder.
  4) Oznitelikler: egitimle BIREBIR ayni fonksiyon (extract_features.compute_features),
     ayni birim donusumu (counts / kazanc = m/s), ayni bant ve pencereler.
     P ve S varislari KOERI analistlerinin elle okumalari.

Ciktilar (config.KOERI altinda):
    stationxml/KO.<ist>.xml, waveforms/<deprem_kodu>.mseed,
    turkey_index.csv, turkey_features.csv.gz, turkey_failed.csv

Kullanim:
    Spyder: F5 (tum adimlar; ayarlar config.py'den)
    Yalnizca bir adim: runfile(..., args='--step download') veya '--step features'
"""
import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

from common import cfg
from extract_features import FS, compute_features

BAND_PRIORITY = ["HH", "BH", "EH"]          # egitimdeki hiz sensorleri
COMP_MAP = {"Z": 0, "N": 1, "1": 1, "E": 2, "2": 2}
PRE_S, POST_S = 20.0, 40.0                  # P'ye gore indirme penceresi
MAX_DIST_KM = 110
MAX_PER_EVENT = 5
CLIP_COUNTS = 0.95 * 2 ** 23                # 24 bit kayitcida kirpilma esigi

_local = threading.local()


def client():
    if not hasattr(_local, "c"):
        _local.c = Client("KOERI", timeout=90)
    return _local.c


def to_utc(x):
    return UTCDateTime(pd.Timestamp(x).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%f"))


# ------------------------------------------------------------------ 1) envanter
def fetch_inventory(stations, xml_dir):
    os.makedirs(xml_dir, exist_ok=True)
    found = {}
    for sta in stations:
        path = os.path.join(xml_dir, f"KO.{sta}.xml")
        if not os.path.exists(path):
            for attempt in range(4):
                try:
                    inv = client().get_stations(network="KO", station=sta, channel="HH?,BH?,EH?",
                                                level="response")
                    inv.write(path, format="STATIONXML")
                    break
                except FDSNNoDataException:
                    break
                except Exception as e:       # or. 502 Bad gateway: bekleyip tekrar dene
                    if attempt == 3:
                        print(f"  {sta}: envanter alinamadi ({str(e)[:60]}); tekrar F5 ile denenir")
                    time.sleep(10 * (attempt + 1))
            if not os.path.exists(path):
                continue
        found[sta] = read_inventory(path)
    return found


def bands_of(inv):
    return {ch.code[:2] for net in inv for st in net for ch in st}


# ------------------------------------------------------------------ 2) secim
def select_pairs(koeri_dir, inventories):
    ev = pd.read_csv(os.path.join(koeri_dir, "koeri_events.csv"), dtype={"event_id": str})
    fa = pd.read_csv(os.path.join(koeri_dir, "koeri_first_arrivals.csv"), dtype={"event_id": str})
    fa = fa[(fa["dist_km"] <= MAX_DIST_KM) & fa["time_P"].notna()]
    usable = {s for s, inv in inventories.items() if bands_of(inv) & set(BAND_PRIORITY)}
    fa = fa[fa["station"].isin(usable)]
    fa = fa.sort_values("dist_km").groupby("event_id").head(MAX_PER_EVENT)
    keep = ["event_id", "origin_time", "lat", "lon", "depth_km", "mag", "catalog_ML"]
    pairs = fa.merge(ev[[c for c in keep if c in ev.columns]], on="event_id", how="inner")
    return pairs.rename(columns={"mag": "ML_bulletin"})


# ------------------------------------------------------------------ 3) indirme
def best_band(inv):
    have = bands_of(inv)
    return next(b for b in BAND_PRIORITY if b in have)


def attempted_stations(path):
    """Bu deprem icin daha once istenmis istasyonlar (isaret dosyasi ya da MiniSEED icerigi)."""
    marker = path + ".stations"
    if os.path.exists(marker):
        return set(open(marker).read().split())
    if os.path.exists(path):          # eski surumle inmis dosya: icerigine bak
        try:
            got = {tr.stats.station for tr in read(path, headonly=True)}
        except Exception:
            got = set()
        with open(marker, "w") as fh:
            fh.write("\n".join(sorted(got)))
        return got
    return set()


def download_event(event_id, rows, wf_dir, inventories):
    """Yalnizca daha once istenmemis istasyonlari indirir ve mevcut dosyaya ekler.
    Her istasyon icin envanterdeki en oncelikli bant istenir (HH > BH > EH)."""
    path = os.path.join(wf_dir, f"{event_id}.mseed")
    done = attempted_stations(path)
    need = [r for r in rows.itertuples() if r.station not in done]
    if not need:
        return event_id, "CACHED"
    bulk = []
    for r in need:
        tp = to_utc(r.time_P)
        bulk.append(("KO", r.station, "*", f"{best_band(inventories[r.station])}?",
                     tp - PRE_S, tp + POST_S))
    err = ""
    for attempt in range(3):
        try:
            try:
                st = client().get_waveforms_bulk(bulk)
            except FDSNNoDataException:
                st = Stream()
            if os.path.exists(path):
                st = read(path) + st
            if len(st):
                st.write(path, format="MSEED")
            with open(path + ".stations", "w") as fh:
                fh.write("\n".join(sorted(done | {r.station for r in need})))
            return event_id, f"OK:{len(st)}"
        except Exception as e:
            err = str(e)[:80]
            time.sleep(5 * (attempt + 1))
    return event_id, f"HATA:{err}"


# ------------------------------------------------------------------ 4) oznitelikler
def sensitivity(inv, seed_id, t):
    try:
        s = inv.get_response(seed_id, t).instrument_sensitivity
        if s is None or str(s.input_units).lower() not in ("m/s", "m/sec"):
            return None
        return abs(s.value)
    except Exception:
        return None


def assemble(st, station, tp, inv):
    """Istasyonun en oncelikli bandindan Z,N,E dizisi (counts) ve kazanclari."""
    for band in BAND_PRIORITY:
        sel = st.select(station=station, channel=f"{band}?")
        if not sel:
            continue
        for loc in sorted({tr.stats.location for tr in sel}):
            s3 = sel.select(location=loc).copy()
            s3.merge(method=1, fill_value=0)
            comps = {}
            for tr in s3:
                c = COMP_MAP.get(tr.stats.channel[-1])
                if c is not None:
                    comps[c] = tr
            if len(comps) < 3:
                continue
            for tr in comps.values():
                if abs(tr.stats.sampling_rate - FS) > 1e-6:
                    tr.resample(FS)
            t0 = max(tr.stats.starttime for tr in comps.values())
            t1 = min(tr.stats.endtime for tr in comps.values())
            if t1 - t0 < 30:
                continue
            arr, sens = [], []
            for c in range(3):
                tr = comps[c].copy().trim(t0, t1)
                arr.append(tr.data[:int((t1 - t0) * FS)].astype(np.float64))
                sens.append(sensitivity(inv, tr.id, tp))
            n = min(len(a) for a in arr)
            if None in sens or n < 30 * FS:
                continue
            return np.vstack([a[:n] for a in arr]), np.array(sens), t0, band, loc
    return None


def compute_event(path, rows, inventories):
    out = []
    try:
        st = read(path)
    except Exception:
        return out
    for r in rows.itertuples():
        tp = to_utc(r.time_P)
        got = assemble(st, r.station, tp, inventories[r.station])
        if got is None:
            continue
        raw, sens, t0, band, loc = got
        p = (tp - t0) * FS
        s = (to_utc(r.time_S) - t0) * FS if isinstance(r.time_S, str) else np.nan
        f = compute_features(raw, sens, p, s)
        if f is None:
            continue
        f.update(event_id=r.event_id, station=r.station, network="KO", band=band, location=loc,
                 distance_km=r.dist_km, depth_km=r.depth_km, origin_time=r.origin_time,
                 lat=r.lat, lon=r.lon, ML_bulletin=r.ML_bulletin,
                 catalog_ML=getattr(r, "catalog_ML", np.nan), sta_ml=r.sta_ml,
                 has_S=isinstance(r.time_S, str),
                 clipped=bool(np.abs(raw).max() >= CLIP_COUNTS))
        out.append(f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--koeri", default=cfg("KOERI"), required=cfg("KOERI") is None)
    ap.add_argument("--step", choices=["all", "download", "features"], default="all")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    kd = args.koeri
    wf_dir = os.path.join(kd, "waveforms")
    os.makedirs(wf_dir, exist_ok=True)

    fa = pd.read_csv(os.path.join(kd, "koeri_first_arrivals.csv"), dtype={"event_id": str})
    stations = sorted(fa.loc[fa["dist_km"] <= MAX_DIST_KM, "station"].unique())
    print(f"1) Istasyon envanteri ({len(stations)} aday)...", flush=True)
    inventories = fetch_inventory(stations, os.path.join(kd, "stationxml"))
    print(f"   KO agi envanteri bulunan: {len(inventories)}")

    pairs = select_pairs(kd, inventories)
    pairs.to_csv(os.path.join(kd, "turkey_index.csv"), index=False)
    groups = dict(tuple(pairs.groupby("event_id")))
    print(f"2) Secim: {len(pairs):,} deprem-istasyon cifti, {len(groups):,} deprem", flush=True)

    if args.step in ("all", "download"):
        print(f"3) Indirme ({args.threads} paralel istek)...", flush=True)
        todo = [e for e in groups
                if not set(groups[e]["station"]) <= attempted_stations(os.path.join(wf_dir, f"{e}.mseed"))]
        print(f"   Indirilecek/tamamlanacak: {len(todo):,} deprem "
              f"({len(groups) - len(todo):,} deprem tamam)")
        failed, t0 = [], time.time()
        with ThreadPoolExecutor(args.threads) as ex:
            futs = [ex.submit(download_event, e, groups[e], wf_dir, inventories) for e in todo]
            for i, f in enumerate(as_completed(futs), 1):
                e, status = f.result()
                if status.startswith("HATA"):
                    failed.append((e, status))
                if i % 200 == 0 or i == len(futs):
                    rate = i / max(time.time() - t0, 1)
                    print(f"   {i:,}/{len(futs):,} deprem, ~{(len(futs) - i) / max(rate, 1e-6) / 60:.0f} dk kaldi",
                          flush=True)
        pd.DataFrame(failed, columns=["event_id", "status"]).to_csv(
            os.path.join(kd, "turkey_failed.csv"), index=False)
        print(f"   Basarisiz: {len(failed)} (tekrar F5 ile yeniden denenir)")

    if args.step in ("all", "features"):
        print("4) Oznitelikler...", flush=True)
        rows = []
        for i, (e, g) in enumerate(groups.items(), 1):
            path = os.path.join(wf_dir, f"{e}.mseed")
            if os.path.exists(path):
                rows += compute_event(path, g, inventories)
            if i % 1000 == 0:
                print(f"   {i:,}/{len(groups):,} deprem islendi", flush=True)
        feats = pd.DataFrame(rows)
        feats.insert(0, "dataset", "TURKIYE_SE")
        out = os.path.join(kd, "turkey_features.csv.gz")
        feats.to_csv(out, index=False)
        print(f"\nOzniteligi hesaplanan iz: {len(feats):,} / {len(pairs):,} cift, "
              f"{feats['event_id'].nunique():,} deprem")
        if len(feats):
            print(f"Kirpilmis (clipped) iz: {feats['clipped'].sum():,} | S varisi olan: {feats['has_S'].mean():.1%}")
            print("Bant dagilimi:", feats["band"].value_counts().to_dict())
            print("ML (bulten) dagilimi:")
            print(pd.cut(feats["ML_bulletin"], [0, 1, 2, 3, 4, 5, 10]).value_counts().sort_index().to_string())
            print("snr_db ozeti:", feats["snr_db"].describe().round(1).to_dict())
        print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()