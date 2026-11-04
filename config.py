"""
PROJE AYARLARI - tum scriptler bu dosyadan okur.
Spyder'da F5 (Run) ile calistirildiginda argumana gerek kalmaz.
Baska bir makinede yalnizca bu dosyadaki yollari degistirin.
Komut satirindan verilen argumanlar (or. --root ...) bu degerlerin yerine gecer.
"""
# Klasorler
PROJECT = "D:/sibel/deprem"
ROOT = PROJECT + "/DataBase"            # SeisBench veri setleri, oznitelikler, sonuclar
KOERI = PROJECT + "/KOERI"              # KOERI katalog + faz okumalari
KOERI_FAZ = KOERI + "/faz"
KOERI_CATALOG = KOERI + "/20230206_20261026_2.5_9.0_23_948.txt"

# Veri setleri ve yillar
DATASETS = ["STEAD", "INSTANCE", "CEED"]
KOERI_YEARS = [2023, 2024, 2025, 2026]

# Kalite filtresi: tum veri setlerinde AYNI tanimla, dalga formundan hesaplanan SNR
# (extract_features.py'deki snr_db = 20*log10(PGV / gurultu RMS), 1-20 Hz, m/s)
MIN_SNR_DB = 20.0

# Hesaplama
WORKERS = 8          # oznitelik cikariminda paralel surec sayisi
N_JOBS = 16          # model egitiminde is parcacigi sayisi
SEED = 42