"""Scriptlerin ortak yardimci fonksiyonlari."""
import os

# Windows'ta PyTorch (SeisBench) ve Intel MKL OpenMP kutuphanelerinin cakismasini onler.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path

SB_NAMES = {"STEAD": "STEAD", "INSTANCE": "InstanceCounts", "CEED": "CEED"}

try:
    import config as C
except ImportError:      # config.py yoksa (or. baska sunucu) yalnizca komut satiri kullanilir
    C = None


def cfg(name, fallback=None):
    """config.py'deki degeri dondurur; yoksa fallback."""
    return getattr(C, name, fallback) if C is not None else fallback


def set_seisbench_root(root):
    """SeisBench veri konumunu ayarlar; SeisBench onceden yuklenmis olsa bile calisir
    (Spyder'da ayni konsolda birden fazla script calistirildiginda gerekli)."""
    os.environ["SEISBENCH_CACHE_ROOT"] = str(root)
    import seisbench
    seisbench.cache_root = Path(root)
    seisbench.cache_data_root = Path(root) / "datasets"


def load_sb_dataset(name, root):
    """SeisBench veri setini acar. CEED icin parca listesi yerel 'chunks' dosyasindan okunur."""
    set_seisbench_root(root)
    import seisbench.data as sbd
    if name == "CEED":
        chunks = open(Path(root) / "datasets" / "ceed" / "chunks").read().split()
        return sbd.CEED(chunks=chunks)
    return getattr(sbd, SB_NAMES[name])()