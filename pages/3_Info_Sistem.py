import hashlib
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd
import streamlit as st

from core import storage, ui_theme

storage.ensure_dirs()

ROOT = Path(__file__).resolve().parents[1]

# Nama paket sesuai nama di pip (bukan nama import)
PAKET = [
    "torch", "numpy", "ultralytics", "supervision", "onnxruntime",
    "opencv-python", "pandas", "insightface", "openpyxl", "psutil",
    "streamlit", "imageio-ffmpeg",
]


def versi_paket(nama):
    try:
        return metadata.version(nama)
    except metadata.PackageNotFoundError:
        return "tidak terpasang"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blok)
    return h.hexdigest().upper()


ui_theme.page_header(
    "Info sistem",
    "Versi perangkat lunak dan sidik file (SHA256) untuk dokumentasi reproduksibilitas.",
)

# ---------- Lingkungan ----------
ui_theme.section("Lingkungan")
lingk = [
    ("Python", platform.python_version()),
    ("Interpreter", sys.executable),
    ("Sistem operasi", platform.platform()),
    ("Prosesor", platform.processor() or "tidak terbaca"),
    ("Jumlah core logis", str(os.cpu_count())),
]
try:
    import psutil
    lingk.append(("RAM total (GB)", f"{psutil.virtual_memory().total / 1024**3:.1f}"))
except Exception:
    pass
df_lingk = pd.DataFrame(lingk, columns=["Item", "Nilai"])
st.dataframe(df_lingk, hide_index=True, width="stretch")

if "cctv_streamlit" not in sys.executable.lower():
    st.warning("Interpreter ini tampaknya bukan venv proyek cctv_streamlit. "
               "Versi di bawah mungkin bukan versi yang dipakai untuk data skripsi.")

# ---------- Paket ----------
ui_theme.section("Versi library")
df_paket = pd.DataFrame([(n, versi_paket(n)) for n in PAKET], columns=["Paket", "Versi"])
st.dataframe(df_paket, hide_index=True, width="stretch")

# ---------- Hash file ----------
ui_theme.section("Sidik file (SHA256)")
daftar = [ROOT / "app.py", ROOT / "yolov8n.pt", ROOT / "workspace" / "gallery.pkl"]
daftar += sorted((ROOT / "core").glob("*.py"))
daftar += sorted((ROOT / "pages").glob("*.py"))

baris = []
for p in daftar:
    if p.exists():
        baris.append((str(p.relative_to(ROOT)), p.stat().st_size, sha256_file(p)))
    else:
        baris.append((str(p.relative_to(ROOT)), 0, "FILE TIDAK ADA"))
df_hash = pd.DataFrame(baris, columns=["File", "Ukuran (byte)", "SHA256"])
st.dataframe(df_hash, hide_index=True, width="stretch")

# ---------- Unduh ----------
ui_theme.section("Unduh")
gabung = pd.concat([
    df_lingk.rename(columns={"Item": "Nama", "Nilai": "Nilai"}).assign(Kelompok="Lingkungan"),
    df_paket.rename(columns={"Paket": "Nama", "Versi": "Nilai"}).assign(Kelompok="Library"),
    df_hash.rename(columns={"File": "Nama", "SHA256": "Nilai"}).drop(columns=["Ukuran (byte)"]).assign(Kelompok="SHA256"),
])[["Kelompok", "Nama", "Nilai"]]
st.download_button("Unduh info_sistem.csv",
                   gabung.to_csv(index=False).encode("utf-8"),
                   file_name="info_sistem.csv", mime="text/csv")
