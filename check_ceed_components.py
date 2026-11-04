"""
CEED bilesen sirasi kontrolu.
SeisBench CEED icin bilesen sirasini tanimli gostermiyor ('None'). Bu script:
  1) metadata'da bilesen/kanal bilgisi iceren sutunlari listeler,
  2) rastgele 500 depremli izde P varisi etrafindaki enerjiyi kanal bazinda
     karsilastirir. Dusey (Z) bilesen, P dalgasinin ilk 1 saniyesinde yataylara
     gore belirgin bicimde daha fazla enerji tasir; bu fiziksel ozellikten
     hangi indeksin Z oldugu cikarilir.

Kullanim:
    python check_ceed_components.py --root D:/sibel/deprem/DataBase
"""
import argparse
import os

import numpy as np

from common import cfg, load_sb_dataset


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=cfg("ROOT"), required=cfg("ROOT") is None)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args(argv)
    ds = load_sb_dataset("CEED", args.root)
    md = ds.metadata
    print(f"CEED: {len(md):,} iz, {len(md.columns)} sutun")
    print("\nTum sutunlar:\n" + ", ".join(md.columns))

    keys = ("component", "channel", "orient", "instrument", "unit", "sampling", "magnitude_type")
    print("\nBilesen/kanal ile ilgili sutunlar:")
    for c in md.columns:
        if any(k in c.lower() for k in keys):
            print(f"  {c:40s} ornek: {md[c].dropna().astype(str).value_counts().head(5).to_dict()}")

    pcol = next((c for c in md.columns if c.lower().startswith("trace_p") and "sample" in c.lower()), None)
    if pcol is None:
        print("\nP varis sutunu bulunamadi; enerji testi atlandi.")
        return
    cand = md[md[pcol].notna()].sample(min(args.n, md[pcol].notna().sum()), random_state=0)
    ratios = []
    for idx, row in cand.iterrows():
        w = ds.get_waveforms(int(idx)).astype(float)
        p = int(row[pcol])
        fs = 100
        if p < 2 * fs or p + fs > w.shape[1]:
            continue
        noise = w[:, p - 2 * fs:p - fs]
        sig = w[:, p:p + fs]
        e = (sig - noise.mean(axis=1, keepdims=True)).std(axis=1) / (noise.std(axis=1) + 1e-9)
        if np.all(np.isfinite(e)) and e.sum() > 0:
            ratios.append(e / e.sum())
    r = np.array(ratios)
    print(f"\nP'nin ilk 1 s'sindeki goreli enerji payi (kanal indeksi 0, 1, 2), {len(r)} iz:")
    print(f"  medyan: {np.round(np.median(r, axis=0), 3)}")
    print(f"  en yuksek enerjili kanal sikligi: {np.bincount(r.argmax(axis=1), minlength=3)}")
    print("\nYorum: Bir indeks belirgin bicimde one cikiyorsa (or. pay ~0.45+), Z bilesen odur.")
    print(f"SeisBench'in bildirdigi cikis sirasi: {ds.component_order}")


if __name__ == "__main__":
    main()
