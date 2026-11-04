"""
Birinci makalenin tum deneylerini sirayla calistirir (Spyder: F5).
Her deney kendi klasorune kaydedilir; tamamlanmis deneyler ATLANIR, bu yuzden
yarida kalirsa tekrar F5'e basmak yeterli.

Varsayilan (F5): ana deney + uc LODO deneyi + bolgeler arasi ensemble ayari.
S-P'siz alt modeller icin: runfile(..., args='--set nosp')
Ablasyonlar icin: runfile(..., args='--set ablations')
"""
import os

# Windows: farkli paketlerin getirdigi iki OpenMP kutuphanesi ayni surece yuklenince Python
# cokebiliyor ("OMP: Error #15"). numpy'den ONCE ayarlanmali.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse

import train_models
import tune_ensembles
from common import cfg

ALL = cfg("DATASETS", ["STEAD", "INSTANCE", "CEED"])

CORE = [
    # (aciklama, train_models argumanlari)
    ("Ana deney (tum veri setleri)", ["--train-datasets", *ALL]),
    *[(f"LODO: yalnizca {d}", ["--train-datasets", d]) for d in ALL],
]
# S-P'siz alt modeller: S varisi olmayan izlere uygulanir (pattern submodel)
NOSP = [(f"{d} (S-P'siz)", a + ["--exclude-features", "sp_time_s"]) for d, a in CORE]

ABLATIONS = [
    ("Ablasyon: dogal dagilim", ["--train-datasets", *ALL, "--cap", "0"]),
    ("Ablasyon: katalog uzakligi", ["--train-datasets", *ALL, "--with-distance"]),
    ("Ablasyon: yalnizca agac modelleri", ["--train-datasets", *ALL, "--pool", "trees"]),
    ("Ablasyon: zamana gore bolme", ["--train-datasets", *ALL, "--split-mode", "time"]),
]
EXPERIMENTS = CORE + NOSP + ABLATIONS


def result_dir(args):
    """train_models.py'nin klasor adlandirmasini taklit eder (tamamlanmis deneyi atlamak icin)."""
    a = dict(datasets=ALL, cap="40000", split="event", pool="diverse", dist="nodist", excl=[])
    it = iter(args)
    for k in it:
        if k == "--train-datasets":
            a["datasets"] = []
            continue
        if k == "--cap":
            a["cap"] = next(it)
        elif k == "--split-mode":
            a["split"] = next(it)
        elif k == "--pool":
            a["pool"] = next(it)
        elif k == "--with-distance":
            a["dist"] = "dist"
        elif k == "--exclude-features":
            a["excl"] = [next(it)]
        elif not k.startswith("--") and isinstance(a["datasets"], list) and a["excl"] == []:
            a["datasets"].append(k)
    tag = (f"train-{'+'.join(a['datasets'])}_cap{a['cap']}_{a['split']}_{a['pool']}_"
           f"{a['dist']}{'_excl-' + '-'.join(a['excl']) if a['excl'] else ''}_seed{cfg('SEED', 42)}")
    return os.path.join(cfg("ROOT"), "results", tag)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["core", "nosp", "ablations", "all"], default="core",
                    help="core: ana deney + LODO + ensemble ayari; nosp: S-P'siz alt modeller; ablations; all")
    a = ap.parse_args(argv)
    exps = {"core": CORE, "nosp": NOSP, "ablations": ABLATIONS, "all": EXPERIMENTS}[a.set]
    for i, (desc, args) in enumerate(exps, 1):
        out = result_dir(args)
        if os.path.exists(os.path.join(out, "metrics_test.csv")):
            print(f"[{i}/{len(exps)}] {desc}: zaten tamam, atlandi")
            continue
        print(f"\n{'=' * 70}\n[{i}/{len(exps)}] {desc}\n{'=' * 70}", flush=True)
        train_models.main(args)
    if a.set in ("core", "all"):
        print(f"\n{'=' * 70}\nBolgeler arasi ensemble ayari (LODO deneyleri)\n{'=' * 70}", flush=True)
        tune_ensembles.main([])


if __name__ == "__main__":
    main()