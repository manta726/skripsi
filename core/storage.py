import hashlib
import os
import pickle
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
GALLERY = WORKSPACE / "gallery.pkl"
UPLOADS = WORKSPACE / "uploads"
OUTPUTS = WORKSPACE / "outputs"
BACKUPS = WORKSPACE / "backups"
ENROLL_PHOTOS = WORKSPACE / "enrollment"  # foto asli per karyawan
RUN_STATE = "hasil.pkl"  # ringkasan satu run, dibaca ulang saat halaman dibuka
RUN_XLSX = "hasil_evaluasi.xlsx"


def ensure_dirs():
    for d in (WORKSPACE, UPLOADS, OUTPUTS, BACKUPS, ENROLL_PHOTOS):
        d.mkdir(parents=True, exist_ok=True)


def load_raw_db():
    """{user: [embedding, ...]}; kosong bila belum ada."""
    if GALLERY.exists():
        with open(GALLERY, "rb") as f:
            return pickle.load(f)
    return {}


def save_raw_db(db):
    ensure_dirs()
    tmp = GALLERY.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(db, f)
    os.replace(tmp, GALLERY)


def sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "?"


def backup_gallery():
    ensure_dirs()
    dst = BACKUPS / f"gallery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    shutil.copy2(GALLERY, dst)
    return dst


def save_upload(uploaded):
    """Simpan file unggahan sekali saja (nama berawalan ukuran agar tidak bentrok)."""
    ensure_dirs()
    size = uploaded.size
    p = UPLOADS / f"{size}_{uploaded.name}"
    if not p.exists() or p.stat().st_size != size:
        with open(p, "wb") as f:
            f.write(uploaded.getbuffer())
    return p


def new_run_dir():
    ensure_dirs()
    d = OUTPUTS / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------- Foto enrollment ----------------

def save_enroll_photos(uid, files, replace=False):
    """Simpan foto asli karyawan agar bisa ditampilkan lagi setelah aplikasi ditutup."""
    ensure_dirs()
    d = ENROLL_PHOTOS / uid
    if replace and d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    paths = []
    for i, fl in enumerate(files):
        nama = re.sub(r"[^A-Za-z0-9._-]", "_", fl.name)
        p = d / f"{stamp}_{i:02d}_{nama}"
        p.write_bytes(fl.getvalue())
        paths.append(p)
    return paths


def list_enroll_photos(uid):
    d = ENROLL_PHOTOS / uid
    return sorted(d.iterdir()) if d.exists() else []


def delete_enroll_photos(uid):
    d = ENROLL_PHOTOS / uid
    if d.exists():
        shutil.rmtree(d)


# ---------------- Hasil proses (run) ----------------

def save_run_state(run_dir, runs, cfg_df, excel_bytes):
    """Simpan ringkasan run dan Excel di folder run, berdampingan dengan video anotasi."""
    with open(run_dir / RUN_STATE, "wb") as f:
        pickle.dump({"runs": runs, "cfg_df": cfg_df}, f)
    (run_dir / RUN_XLSX).write_bytes(excel_bytes)


def list_saved_runs():
    """Folder run yang punya ringkasan, terbaru di atas."""
    if not OUTPUTS.exists():
        return []
    dirs = [d for d in OUTPUTS.iterdir() if d.is_dir() and (d / RUN_STATE).exists()]
    return sorted(dirs, key=lambda d: d.name, reverse=True)


def load_run_state(run_dir):
    with open(Path(run_dir) / RUN_STATE, "rb") as f:
        return pickle.load(f)
