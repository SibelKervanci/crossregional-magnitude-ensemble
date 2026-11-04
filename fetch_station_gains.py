"""
Istasyon kazanc (sensitivity) bilgilerini FDSN veri merkezlerinden indirir.
Amac: counts -> fiziksel birim (m/s) donusumu icin harmonizasyon.

Kaldigi yerden devam eder: indirilen StationXML dosyalari atlanir.

Kullanim:
    nohup python fetch_station_gains.py --root ~/work/deprem/DataBase > gains.log 2>&1 &
"""
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from obspy import read_inventory
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from common import cfg


VEL_CHANNELS = ["HH", "BH", "EH"]   # yalnizca hiz sensorleri
MAX_DIST_KM = 110
PROVIDERS = {"STEAD": ["IRIS", "NCEDC", "SCEDC"], "INSTANCE": ["INGV", "IRIS"]}
FALLBACK = ["IRIS", "NCEDC", "SCEDC", "INGV", "ORFEUS", "GFZ", "ETH"]

_local = threading.local()


def get_client(name):
    """Her thread icin ayri FDSN istemcisi (thread guvenligi icin)."""
    if not hasattr(_local, "clients"):
        _local.clients = {}
    if name not in _local.clients:
        _local.clients[name] = Client(name, timeout=60, _discover_services=False)
    return _local.clients[name]


def stations_stead(base):
    cols = ["station_network_code", "station_code", "trace_channel",
            "source_magnitude_type", "source_distance_km"]
    md = pd.read_csv(os.path.join(base, "stead", "metadata.csv"), usecols=cols, low_memory=False)
    md = md[(md["source_magnitude_type"].astype(str).str.lower() == "ml")
            & (md["source_distance_km"] <= MAX_DIST_KM)
            & (md["trace_channel"].isin(VEL_CHANNELS))]
    print(f"STEAD secilen iz: {len(md):,}")
    md = md.rename(columns={"trace_channel": "channel"})
    return md[["station_network_code", "station_code", "channel"]].assign(dataset="STEAD")


def stations_instance(base):
    cols = ["station_network_code", "station_code", "station_channels",
            "source_magnitude_type", "path_ep_distance_km"]
    md = pd.read_csv(os.path.join(base, "instancecounts", "metadata.csv"), usecols=cols, low_memory=False)
    md = md[(md["source_magnitude_type"].astype(str).str.lower() == "ml")
            & (md["path_ep_distance_km"] <= MAX_DIST_KM)
            & (md["station_channels"].isin(VEL_CHANNELS))]
    print(f"INSTANCE secilen iz: {len(md):,}")
    md = md.rename(columns={"station_channels": "channel"})
    return md[["station_network_code", "station_code", "channel"]].assign(dataset="INSTANCE")


def fetch(net, sta, channels, datasets, xml_dir):
    out = os.path.join(xml_dir, f"{net}.{sta}.xml")
    if os.path.exists(out):
        return net, sta, "CACHED"
    providers = []
    for ds in datasets:
        providers += PROVIDERS[ds]
    providers = list(dict.fromkeys(providers + FALLBACK))  # sirayi koruyarak tekillestir
    chan_query = ",".join(f"{c}?" for c in sorted(channels))
    for prov in providers:
        try:
            inv = get_client(prov).get_stations(network=net, station=sta,
                                                channel=chan_query, level="response")
            inv.write(out, format="STATIONXML")
            return net, sta, prov
        except FDSNNoDataException:
            continue
        except Exception:
            continue
    return net, sta, "FAILED"


def build_gain_table(xml_dir, out_csv):
    rows = []
    for fname in sorted(os.listdir(xml_dir)):
        if not fname.endswith(".xml"):
            continue
        inv = read_inventory(os.path.join(xml_dir, fname))
        for net in inv:
            for sta in net:
                for ch in sta:
                    sens = ch.response.instrument_sensitivity if ch.response else None
                    rows.append({
                        "network": net.code, "station": sta.code,
                        "location": ch.location_code, "channel": ch.code,
                        "start": ch.start_date, "end": ch.end_date,
                        "sensitivity": sens.value if sens else None,
                        "input_units": sens.input_units if sens else None,
                        "sens_freq_hz": sens.frequency if sens else None,
                    })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Kazanc tablosu: {out_csv} ({len(rows):,} kanal donemi)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=cfg("ROOT"), required=cfg("ROOT") is None)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    root = os.path.expanduser(args.root)
    base = os.path.join(root, "datasets")
    xml_dir = os.path.join(root, "stationxml")
    os.makedirs(xml_dir, exist_ok=True)

    st = pd.concat([stations_stead(base), stations_instance(base)])
    grouped = st.groupby(["station_network_code", "station_code"]).agg(
        channels=("channel", lambda x: set(x)), datasets=("dataset", lambda x: set(x))
    ).reset_index()
    print(f"Indirilecek benzersiz istasyon: {len(grouped):,}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch, r.station_network_code, r.station_code,
                             r.channels, r.datasets, xml_dir)
                   for r in grouped.itertuples()]
        for i, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            if i % 100 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} istasyon islendi", flush=True)

    res = pd.DataFrame(results, columns=["network", "station", "source"])
    print("\nKaynaklara gore sonuc:")
    print(res["source"].value_counts().to_string())
    failed = res[res["source"] == "FAILED"]
    failed.to_csv(os.path.join(root, "failed_stations.csv"), index=False)
    print(f"Basarisiz istasyon: {len(failed):,} (failed_stations.csv)")

    build_gain_table(xml_dir, os.path.join(root, "station_gains.csv"))


if __name__ == "__main__":
    main()
