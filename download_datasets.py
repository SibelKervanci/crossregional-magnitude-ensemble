"""
Veri seti indirme ve ilk inceleme scripti
Hedef: Ensemble tabanli deprem buyuklugu tahmini (Computers & Geosciences)

Kurulum:
    pip install seisbench huggingface_hub pandas

Kullanim:
    python download_datasets.py --root /veri/seisbench --stage 1   # STEAD + INSTANCE (~230 GB)
    python download_datasets.py --root /veri/seisbench --stage 2   # CEED (~577 GB)
"""
import argparse
import os
import shutil
import sys

# Boyutlar SeisBench dokumantasyonundan (GB)
DATASETS = {
    1: {"STEAD": 70.2, "InstanceCounts": 160},
    2: {"CEED": 577},
}


def check_disk(root, needed_gb):
    os.makedirs(root, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / 1e9
    print(f"Bos alan: {free_gb:.0f} GB | Gereken: ~{needed_gb:.0f} GB (+%10 pay)")
    if free_gb < needed_gb * 1.1:
        sys.exit("Yetersiz disk alani. Baska bir --root dizini secin.")


def summarize(name, ds):
    """Makale icin kritik metadata ozetini yazdirir."""
    md = ds.metadata
    print(f"\n===== {name} =====")
    print(f"Toplam iz sayisi: {len(md):,}")

    if "source_id" in md.columns:
        print(f"Benzersiz olay sayisi: {md['source_id'].nunique():,}")

    if "trace_sampling_rate_hz" in md.columns:
        print("Ornekleme hizlari:", md["trace_sampling_rate_hz"].value_counts().to_dict())

    if "source_magnitude_type" in md.columns:
        print("Buyukluk turleri:")
        print(md["source_magnitude_type"].value_counts(dropna=False).head(10).to_string())

    if "source_magnitude" in md.columns:
        print("Buyukluk dagilimi:")
        print(md["source_magnitude"].describe().round(2).to_string())

    if "split" in md.columns:
        print("Hazir bolme:", md["split"].value_counts().to_dict())

    # Ozet CSV: harmonizasyon asamasinda kullanilacak
    out = os.path.join(os.environ["SEISBENCH_CACHE_ROOT"], f"{name}_metadata_summary.csv")
    md.head(1000).to_csv(out, index=False)
    print(f"Ilk 1000 satir kaydedildi: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Veri setlerinin indirilecegi dizin")
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--backup", action="store_true",
                   help="SeisBench yedek deposunu kullan (ana depo erisilemezse)")
    args = p.parse_args()

    # SeisBench bu ortam degiskenini kullanir; import'tan ONCE ayarlanmali
    os.environ["SEISBENCH_CACHE_ROOT"] = os.path.abspath(args.root)
    import seisbench
    import seisbench.data as sbd
    if args.backup:
        seisbench.use_backup_repository()
        print("SeisBench yedek deposu kullaniliyor.")

    targets = DATASETS[args.stage]
    check_disk(args.root, sum(targets.values()))

    for name in targets:
        print(f"\n>>> {name} indiriliyor (ilk calistirmada uzun surer)...")
        ds = getattr(sbd, name)()  # Ilk cagrida indirme baslar, sonra cache'ten okur
        summarize(name, ds)

    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
