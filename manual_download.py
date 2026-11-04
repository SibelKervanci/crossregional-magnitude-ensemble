"""
SeisBench veri setlerini dogrudan indirir (SeisBench'in kendi indiricisini atlar).
Dosyalar SeisBench'in bekledigi klasor yapisina yazilir; sonra sbd.STEAD() vb.
dosyalari onbellekte bulur ve tekrar indirmez.

Kaldigi yerden devam eder: yarim dosyalar ".part" olarak tutulur; komut tekrar
calistirildiginda kaldigi byte'tan devam edilir.

Kullanim (Windows, Anaconda Prompt):
    python manual_download.py --root D:/sibel/deprem/DataBase --dataset stead
    python manual_download.py --root D:/sibel/deprem/DataBase --dataset instancecounts
    python manual_download.py --root D:/sibel/deprem/DataBase --dataset ceed
"""
import argparse
import os
import sys
import time

import requests

MIRRORS = {
    "main": "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/",
    "backup": "https://seisbench.gfz.de/mirror/",
}
BLOCK = 8 * 1024 * 1024
MAX_RETRIES = 50


def remote_size(url):
    r = requests.head(url, timeout=30, allow_redirects=True)
    if r.status_code != 200:
        return None
    return int(r.headers.get("Content-Length", 0)) or None


def download(url, dest):
    total = remote_size(url)
    if total is None:
        print(f"  [YOK] {url}")
        return False
    if os.path.exists(dest) and os.path.getsize(dest) == total:
        print(f"  [TAMAM, atlandi] {os.path.basename(dest)} ({total / 1e9:.2f} GB)")
        return True
    part = dest + ".part"
    for attempt in range(1, MAX_RETRIES + 1):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if done and r.status_code != 206:      # sunucu devam etmeyi desteklemiyorsa
                    done = 0
                    mode = "wb"
                elif r.status_code in (200, 206):
                    mode = "ab" if done else "wb"
                else:
                    raise IOError(f"HTTP {r.status_code}")
                t0, got, last = time.time(), 0, 0.0
                with open(part, mode) as fh:
                    for blk in r.iter_content(BLOCK):
                        fh.write(blk)
                        got += len(blk)
                        now = time.time()
                        if now - last > 10:
                            cur = done + got
                            speed = got / max(now - t0, 1e-6) / 1e6
                            eta = (total - cur) / max(speed * 1e6, 1) / 60
                            print(f"    {os.path.basename(dest)}: {cur / 1e9:.2f}/{total / 1e9:.2f} GB "
                                  f"({100 * cur / total:.1f}%), {speed:.1f} MB/s, kalan ~{eta:.0f} dk",
                                  flush=True)
                            last = now
            if os.path.getsize(part) == total:
                os.replace(part, dest)
                print(f"  [BITTI] {os.path.basename(dest)}", flush=True)
                return True
            raise IOError("eksik indirme")
        except Exception as e:
            wait = min(60, 5 * attempt)
            print(f"  Hata ({e}); {wait} s sonra devam edilecek (deneme {attempt}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    print(f"  [BASARISIZ] {url}")
    return False


def ceed_chunks(base):
    """CEED parca adlarini bulur. Once depodaki 'chunks' listesine bakar; yoksa
    dosya adlarini yoklar. SeisBench dosya adlari: metadata<parca>.csv, waveforms<parca>.hdf5
    (ornek: metadatanc2010.csv, waveformssc2019_1.hdf5)."""
    r = requests.get(base + "chunks", timeout=30)
    if r.status_code == 200 and r.text.strip():
        return [c.strip() for c in r.text.split() if c.strip()]
    found = []
    for region, years in (("nc", range(1987, 2024)), ("sc", range(1999, 2024))):
        for y in years:
            name = f"{region}{y}"
            if remote_size(f"{base}metadata{name}.csv"):
                found.append(name)
                continue
            i = 1
            while remote_size(f"{base}metadata{name}_{i}.csv"):
                found.append(f"{name}_{i}")
                i += 1
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", required=True, choices=["stead", "instancecounts", "ceed"])
    ap.add_argument("--mirror", choices=list(MIRRORS), default="main")
    ap.add_argument("--chunks", nargs="*", help="Yalnizca CEED: indirilecek parcalar (bos = hepsi)")
    args = ap.parse_args()

    base = f"{MIRRORS[args.mirror]}datasets/{args.dataset}/"
    out = os.path.join(os.path.expanduser(args.root), "datasets", args.dataset)
    os.makedirs(out, exist_ok=True)
    print(f"Kaynak: {base}\nHedef:  {out}", flush=True)

    if args.dataset == "ceed":
        chunks = args.chunks or ceed_chunks(base)
        print(f"CEED parcalari ({len(chunks)}): {' '.join(chunks)}", flush=True)
        if not chunks:
            print("HATA: CEED parcasi bulunamadi.")
            sys.exit(1)
        with open(os.path.join(out, "chunks"), "w") as fh:   # SeisBench yerel parca listesi
            fh.write("\n".join(chunks))
        files = [f for c in chunks for f in (f"metadata{c}.csv", f"waveforms{c}.hdf5")]
    else:
        files = ["metadata.csv", "waveforms.hdf5"]

    ok = all([download(base + f, os.path.join(out, f)) for f in files])
    print("\nTum dosyalar tamam." if ok else "\nBazi dosyalar indirilemedi; komutu tekrar calistirin.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()