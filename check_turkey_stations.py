"""
Turkiye test seti icin istasyon kontrolu:
  1) <=110 km'de en az N okumasi olan istasyonlarin hangi aga ait oldugu
     ve hangi kanallarin (HH/BH/EH) KOERI FDSN servisinde bulundugu,
  2) ornek bir deprem-istasyon cifti icin dalga formu indirme testi.

Kullanim:
    python check_turkey_stations.py --koeri D:/sibel/deprem/KOERI
"""
import argparse
import os

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from common import cfg



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--koeri", default=cfg("KOERI"), required=cfg("KOERI") is None)
    ap.add_argument("--min-picks", type=int, default=50)
    args = ap.parse_args(argv)

    fa = pd.read_csv(os.path.join(args.koeri, "koeri_first_arrivals.csv"))
    near = fa[(fa["dist_km"] <= 110) & fa["time_P"].notna()]
    counts = near["station"].value_counts()
    stations = counts[counts >= args.min_picks].index.tolist()
    print(f"<=110 km ve >= {args.min_picks} okumasi olan istasyon: {len(stations)}")

    client = Client("KOERI", timeout=120)
    rows = []
    for sta in stations:
        try:
            inv = client.get_stations(network="*", station=sta, channel="HH?,BH?,EH?,HN?", level="channel")
            for net in inv:
                for st in net:
                    bands = sorted({ch.code[:2] for ch in st})
                    rows.append({"station": sta, "network": net.code, "bands": ",".join(bands),
                                 "lat": st.latitude, "lon": st.longitude, "n_pairs": counts[sta]})
        except FDSNNoDataException:
            rows.append({"station": sta, "network": "BULUNAMADI", "bands": "", "n_pairs": counts[sta]})
        except Exception as e:
            rows.append({"station": sta, "network": f"HATA:{str(e)[:30]}", "bands": "", "n_pairs": counts[sta]})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.koeri, "turkey_station_check.csv"), index=False)
    print(res.to_string(index=False))

    covered = res[res["bands"].str.contains("HH|BH|EH", na=False)]
    n_cov = near[near["station"].isin(covered["station"])].shape[0]
    print(f"\nHiz sensoru (HH/BH/EH) bulunan istasyon: {covered['station'].nunique()} | "
          f"kapsanan deprem-istasyon cifti: {n_cov:,} / {len(near):,}")

    # Dalga formu indirme testi: en cok okumasi olan istasyondaki ilk cift
    if not covered.empty:
        sta = covered.sort_values("n_pairs", ascending=False).iloc[0]
        pair = near[near["station"] == sta["station"]].iloc[0]
        tp = UTCDateTime(pd.Timestamp(pair["time_P"]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%f"))
        try:
            st = client.get_waveforms(sta["network"], sta["station"], "*", "HH?", tp - 20, tp + 40)
            print(f"\nIndirme testi OK: {sta['network']}.{sta['station']} {tp} -> {len(st)} iz")
            print(st)
        except Exception as e:
            print(f"\nIndirme testi BASARISIZ ({sta['network']}.{sta['station']}): {e}")


if __name__ == "__main__":
    main()
