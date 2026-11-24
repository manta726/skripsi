import os
import re

import cv2
import numpy as np

EYE_RATIO_MIN = 0.15   # di bawah ini dianggap pose profil ekstrem (perkiraan sendiri)


def valid_id(uid):
    return re.fullmatch(r"K\d{3}", uid) is not None


def decode_image(data):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _eye_ratio(face):
    """Jarak horizontal mata / lebar kotak wajah. Mendekati 0 = profil.
    Perkiraan sendiri, bukan rumus yang sama persis dengan register_faces.py."""
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 2:
        return None
    bw = max(float(face.bbox[2] - face.bbox[0]), 1.0)
    return abs(float(kps[1][0] - kps[0][0])) / bw


def embed_from_image(face_app, img):
    """Seperti register_faces.py: wajah dengan det_score tertinggi, tanpa penolakan kualitas."""
    if img is None:
        return None, "File bukan gambar valid", None
    faces = face_app.get(img)
    if not faces:
        return None, "Tidak ada wajah terdeteksi", None
    f = max(faces, key=lambda x: x.det_score)
    notes = []
    if len(faces) > 1:
        notes.append(f"{len(faces)} wajah, dipakai det_score tertinggi")
    if f.det_score < 0.5:
        notes.append("det_score rendah")
    er = _eye_ratio(f)
    if er is not None and er < EYE_RATIO_MIN:
        notes.append(f"pose profil ekstrem (rasio mata={er:.2f}), embedding mungkin kurang akurat")
    return f.embedding.astype(np.float32), "; ".join(notes), float(f.det_score)


def enroll_files(face_app, files):
    """-> (daftar hasil per file, daftar embedding yang berhasil)"""
    results, embs = [], []
    for fl in files:
        img = decode_image(fl.getvalue())
        emb, note, ds = embed_from_image(face_app, img)
        results.append({
            "File": fl.name,
            "Status": "OK" if emb is not None else "GAGAL",
            "det_score": None if ds is None else round(ds, 3),
            "Catatan": note,
        })
        if emb is not None:
            embs.append(emb)
    return results, embs


def add_to_db(db, user, embs, replace=False):
    if replace or user not in db:
        db[user] = list(embs)
    else:
        db[user] = list(db[user]) + list(embs)
    return db


def id_from_filename(name):
    """K001_depan.jpg -> K001. Awalan sebelum '_' dianggap ID karyawan."""
    stem = os.path.splitext(os.path.basename(name))[0]
    return stem.split("_")[0].strip().upper()


def group_files_by_id(files):
    """-> ({ID: [file, ...]}, [nama file yang awalannya bukan ID valid])"""
    groups, invalid = {}, []
    for fl in files:
        uid = id_from_filename(fl.name)
        if valid_id(uid):
            groups.setdefault(uid, []).append(fl)
        else:
            invalid.append(fl.name)
    return groups, invalid
