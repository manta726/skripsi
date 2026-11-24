import hashlib
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
GALLERY = WORKSPACE / "gallery.pkl"
UPLOADS = WORKSPACE / "uploads"
OUTPUTS = WORKSPACE / "outputs"
BACKUPS = WORKSPACE / "backups"


def ensure_dirs():
    for d in (WORKSPACE, UPLOADS, OUTPUTS, BACKUPS):
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