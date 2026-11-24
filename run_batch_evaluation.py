import argparse
import cv2
import os
import sys
import glob
import math
import pickle
import time
import re
import hashlib
import platform
import importlib.metadata
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import supervision as sv
from ultralytics import YOLO
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURATION & PARAMETERS
# ============================================================
DATASET_DIR = "dataset_test"
OUTPUT_DIR = "dataset_test_output"
OUTPUT_EXCEL = "hasil_evaluasi.xlsx"
FACE_DB_PATH = "known_faces_db.pkl"
DEBUG_DIR_ROOT = "debug_faces"

SAVE_ANNOTATED_VIDEOS = False     # True = simpan video berlabel (lebih lambat). Bisa juga via --save
OUTPUT_SCALE = 0.5                # 2304x1296 -> 1152x648
HUD_POSITION = "bottom-left"      # "top-left", "top-right", "bottom-left", "bottom-right"
HUD_SCALE = 0.9

LINE_START = sv.Point(1554, 610)
LINE_END = sv.Point(408, 1146)

RECOGNITION_THRESHOLD = 0.35      # diterapkan per frame yang dinilai; skor yang lolos dijumlahkan per track
YOLO_CONF = 0.3                   # ambang kepercayaan deteksi YOLOv8n (ditetapkan peneliti, tanpa kalibrasi)
YOLO_IMGSZ = 1280
YOLO_DEVICE = "cpu"               # pemrosesan sengaja dibatasi pada CPU
FACE_REC_SKIP_FRAMES = 3          # wajah diproses pada frame kelipatan nilai ini (frame % 3 == 0)
FACE_DET_SIZE = (640, 640)        # ukuran masukan detektor InsightFace

# Ukuran minimum (tinggi dan lebar, piksel) POTONGAN yang diberikan ke InsightFace
# (potongan badan atau kepala, tergantung CROP_MODE). Ini bukan ukuran wajah.
MIN_FACE_SIZE = 30

# Filter gerak: track "bergerak" jika diagonal kotak jelajah titik kepalanya >= nilai ini
# (piksel, resolusi 2304x1296). Nilai awal (tebakan), tidak dikalibrasi pada klip terpisah.
MIN_MOVE_EXTENT = 150

# ---- Mode crop wajah ----
# "body" : seluruh kotak orang, wajah dipilih dengan det_score tertinggi (perilaku lama / baseline)
# "head" : 45% bagian atas kotak orang + pick_own_face (perbaikan logika)
CROP_MODE = "body"
HEAD_FRAC = 0.45                  # tinggi crop kepala = 45% tinggi kotak orang
HEAD_PAD = 0.10                   # lebar crop diperlebar 10% kiri-kanan
HEAD_TOP_PAD = 0.05               # sedikit di atas puncak kotak

# Fallback identitas dari track bergerak lain. Untuk angka utama: MATI.
# Aktifkan hanya dengan --fallback (analisis tambahan).
USE_FALLBACK = False

# Pembagian set (dipakai untuk rekap terpisah)
DEV_DAYS = {"H1", "H2"}           # set pengembangan; selain itu = set uji

# Debug crop
DEBUG_MAX_PER_TRACK = 40          # maksimum gambar wajah per track
DEBUG_OVERVIEW_EVERY = 15         # simpan gambar ikhtisar (ID semua track) tiap N frame


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def parse_ground_truth(rel_path):
    """Ekstrak GT user, aksi, hari dari PATH RELATIF, misal K017/H1/H1_IN.mp4"""
    path_clean = rel_path.replace("\\", "/")

    user_match = re.search(r'(?<![A-Za-z])K\d{3}(?!\d)', path_clean, re.IGNORECASE)
    gt_user = user_match.group(0).upper() if user_match else "UNKNOWN"

    stem = os.path.splitext(os.path.basename(path_clean))[0].upper()
    tokens = [t for t in re.split(r'[_\-\s.]+', stem) if t]
    upper_path = path_clean.upper()
    if any(t in ("IN", "MASUK") for t in tokens) or "/IN/" in upper_path or "/MASUK/" in upper_path:
        gt_action = "MASUK"
    elif any(t in ("OUT", "KELUAR") for t in tokens) or "/OUT/" in upper_path or "/KELUAR/" in upper_path:
        gt_action = "KELUAR"
    else:
        gt_action = "UNKNOWN"

    day_match = re.search(r'(?<![A-Za-z])(H\d+|Hari[_\s]?\d+)', path_clean, re.IGNORECASE)
    gt_day = day_match.group(1).upper() if day_match else "H?"

    return gt_user, gt_action, gt_day


def rel_name(video_path):
    """Path relatif terhadap DATASET_DIR; jika di luar DATASET_DIR, pakai path apa adanya."""
    try:
        rel = os.path.relpath(video_path, DATASET_DIR)
        if rel.startswith(".."):
            rel = video_path
    except ValueError:      # beda drive di Windows
        rel = video_path
    return rel


def reset_tracker(yolo_model):
    """Bersihkan state ByteTrack agar tidak bocor antar video."""
    try:
        if hasattr(yolo_model, 'predictor') and yolo_model.predictor is not None:
            trackers = getattr(yolo_model.predictor, 'trackers', None)
            if trackers:
                for t in trackers:
                    t.reset()
            yolo_model.predictor = None   # paksa dibuat ulang saat track() berikutnya
    except Exception:
        pass


def build_gallery(raw_db):
    """{user: [emb,...]} -> {user: ndarray (N,512) ter-normalisasi L2}"""
    gallery = {}
    for user, embs in raw_db.items():
        if isinstance(embs, np.ndarray) and embs.ndim == 1:
            embs = [embs]
        vecs = []
        for e in embs:
            e = np.asarray(e, dtype=np.float32).flatten()
            n = np.linalg.norm(e)
            if n > 0:
                vecs.append(e / n)
        if vecs:
            gallery[user] = np.stack(vecs)
    return gallery


def match_face(face, gallery):
    """Cocokkan satu wajah ke gallery -> (user_terbaik, cosine_terbaik)."""
    emb = face.embedding.astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    best_sim, best_user = -1.0, "Unknown"
    for user_name, user_embs in gallery.items():
        sim = float(np.max(user_embs @ emb))
        if sim > best_sim:
            best_sim, best_user = sim, user_name
    return best_user, best_sim


class TrackFaceBuffer:
    def __init__(self):
        self.buffer = {}

    def add_recognition(self, track_id, name, score):
        self.buffer.setdefault(track_id, []).append({'name': name, 'score': score})

    def get_final_identity(self, track_id):
        if track_id not in self.buffer or len(self.buffer[track_id]) == 0:
            return "Unknown", 0.0, 0

        votes = self.buffer[track_id]
        user_scores, user_counts = {}, {}
        for item in votes:
            if item['name'] != "Unknown":
                user_scores[item['name']] = user_scores.get(item['name'], 0.0) + item['score']
                user_counts[item['name']] = user_counts.get(item['name'], 0) + 1

        if not user_scores:
            return "Unknown", 0.0, len(votes)

        best_name = max(user_scores, key=user_scores.get)
        return best_name, user_scores[best_name] / user_counts[best_name], user_counts[best_name]


def track_extent(stats):
    """Diagonal kotak jelajah titik kepala sebuah track (piksel). Tahan terhadap jitter."""
    return math.hypot(stats['maxx'] - stats['minx'], stats['maxy'] - stats['miny'])


# ---------------- Crop wajah ----------------
def head_crop(frame, xyxy, frac=HEAD_FRAC, pad=HEAD_PAD, top_pad=HEAD_TOP_PAD):
    """Potongan bagian atas kotak orang (area kepala). None jika degenerate."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 - pad * bw))
    cx2 = int(min(W, x2 + pad * bw))
    cy1 = int(max(0, y1 - top_pad * bh))
    cy2 = int(min(H, y1 + frac * bh))
    if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
        return None
    return frame[cy1:cy2, cx1:cx2]


def pick_own_face(faces, crop_shape):
    """Pilih wajah milik pemilik crop: pusat wajah di tengah crop (x 15-85%, y <= 80%),
    skor = det_score x luas. Mengembalikan None jika tidak ada yang memenuhi."""
    h, w = crop_shape[:2]
    best, best_s = None, -1.0
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
        if not (0.15 <= cx <= 0.85) or cy > 0.8:
            continue
        s = float(f.det_score) * max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if s > best_s:
            best, best_s = f, s
    return best


# ---------------- Debug crop ----------------
def save_debug_face(crop, faces, chosen, tid, frame_no, gallery, out_dir, counters):
    """Simpan crop yang diberikan ke InsightFace. SEMUA wajah digambar beserta kecocokan
    galerinya; hijau = wajah yang dipakai, merah = wajah lain / ditolak."""
    if counters.get(tid, 0) >= DEBUG_MAX_PER_TRACK:
        return
    counters[tid] = counters.get(tid, 0) + 1

    vis = crop.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, vis.shape[1] / 450.0)
    th = max(2, int(round(max(vis.shape[:2]) / 200)))
    chosen_txt = None
    for f in faces:
        b = f.bbox.astype(int)
        user, sim = match_face(f, gallery)
        is_ch = chosen is not None and f is chosen
        color = (0, 255, 0) if is_ch else (0, 0, 255)
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), color, th)
        cv2.putText(vis, f"{user} {sim:.2f} d={f.det_score:.2f}",
                    (b[0], max(18, b[1] - 6)), font, scale * 0.6, color, max(1, th - 1), cv2.LINE_AA)
        if is_ch:
            chosen_txt = f"{user}_{sim:.2f}"
    tag = chosen_txt if chosen_txt else "REJECT"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"T{tid}_f{frame_no:04d}_{tag}.jpg"), vis)


def save_overview(frame, detections, frame_no, out_dir):
    """Gambar ikhtisar: semua track beserta ID-nya (untuk tahu T3 itu siapa)."""
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
        b = xyxy.astype(int)
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (255, 200, 0), 4)
        cv2.putText(vis, f"T{int(tid)}", (b[0], max(40, b[1] - 10)), font, 1.8, (255, 200, 0), 4, cv2.LINE_AA)
    cv2.line(vis, (LINE_START.x, LINE_START.y), (LINE_END.x, LINE_END.y), (0, 255, 255), 4)
    vis = cv2.resize(vis, None, fx=0.5, fy=0.5)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"overview_f{frame_no:04d}.jpg"), vis)


# ---------------- Identitas & ringkasan ----------------
def resolve_identity(face_buffer, track_id, moving_tids):
    """Putuskan identitas di akhir video. Fallback (jika aktif) hanya dari track yang bergerak."""
    name, conf, n = face_buffer.get_final_identity(track_id)
    source = "TRACK"

    if name == "Unknown" and USE_FALLBACK:
        best = None
        for tid in face_buffer.buffer:
            if tid == track_id or tid not in moving_tids:
                continue
            cand = face_buffer.get_final_identity(tid)
            if cand[0] != "Unknown":
                strength = cand[1] * cand[2]
                if best is None or strength > best[0]:
                    best = (strength, cand)
        if best is not None:
            name, conf, n = best[1]
            source = "MOVING_FALLBACK"

    if name == "Unknown":
        source = "NONE"
    return name, conf, n, source


def build_track_summary(track_stats, face_buffer, track_max, crossing_count, moving_tids):
    parts = []
    for tid in sorted(track_stats):
        name, avg, n = face_buffer.get_final_identity(tid)
        mv = "GERAK" if tid in moving_tids else "STATIS"
        mx = track_max.get(tid)
        mx_txt = f"{mx:.2f}" if mx is not None else "-"
        parts.append(f"T{tid}[{mv}] {name} avg={avg:.2f} n={n} max={mx_txt} "
                     f"ext={track_extent(track_stats[tid]):.0f} x={crossing_count.get(tid, 0)}")
    return " | ".join(parts)


def classify_clip(gt_user, gt_action, pred_user, pred_action):
    """Kategori eksklusif (urutan prioritas): no_crossing > unknown > wrong_identity
    > wrong_direction > benar."""
    if pred_action == "NO_CROSSING":
        return "no_crossing"
    if pred_user == "Unknown":
        return "unknown"
    if pred_user != gt_user:
        return "wrong_identity"
    if pred_action != gt_action:
        return "wrong_direction"
    return "benar"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d * 100, (c + a) / d * 100


def peak_ram_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, 'peak_wset', None)       # Windows: puncak sebenarnya
        return round((peak if peak else mi.rss) / 1048576, 1)
    except Exception:
        return None


# ---------------- Berkas keluaran ----------------
def out_paths(excel_path):
    base = excel_path[:-5] if excel_path.endswith(".xlsx") else excel_path
    return {
        'hasil': base + "_backup.csv",
        'obs': base + "_observasi_backup.csv",
        'cross': base + "_crossing_backup.csv",
        'track': base + "_track_backup.csv",
        'err': base + "_error.csv",
    }


def append_df(path, df):
    if df is None or len(df) == 0:
        return
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False, encoding="utf-8-sig")


def done_clips(path):
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path, encoding="utf-8-sig")['Path_Video'].astype(str))
    except Exception:
        return set()


def save_excel_safe(sheets, path):
    """Simpan Excel. Jika file terkunci (masih dibuka di Excel), pakai nama berstempel waktu."""
    def _write(p):
        with pd.ExcelWriter(p, engine='openpyxl') as writer:
            for sheet_name, (frame, keep_index) in sheets.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=keep_index)

    try:
        _write(path)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        print(f"\n[WARN] '{path}' terkunci (tutup di Excel). Menyimpan ke '{alt}'")
        _write(alt)
        return alt


def draw_hud(img, text, position="bottom-left", scale=0.9, thickness=2,
             color=(0, 255, 255), alpha=0.55, margin=15, y_offset=0):
    """Kotak teks kecil semi-transparan, lebarnya mengikuti teks."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    pad = 10
    box_w, box_h = tw + 2 * pad, th + base + 2 * pad

    x0 = margin if "left" in position else w - box_w - margin
    y0 = margin if "top" in position else h - box_h - margin
    y0 += y_offset
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x0 + box_w), min(h, y0 + box_h)

    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0, roi)
    cv2.putText(img, text, (x0 + pad, y0 + pad + th), font, scale, color, thickness, cv2.LINE_AA)


# ---------------- Snapshot konfigurasi ----------------
def sha256_of(path):
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "?"


def pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "?"


def bytetrack_params():
    """Baca bytetrack.yaml bawaan Ultralytics yang benar-benar dipakai."""
    try:
        import ultralytics
        import yaml
        p = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "bytetrack.yaml")
        with open(p, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def build_config_table(gallery, n_videos, debug_on):
    try:
        import torch
        torch_ver, cuda = torch.__version__, torch.cuda.is_available()
    except Exception:
        torch_ver, cuda = "?", "?"

    script_path = os.path.abspath(__file__)
    rows = [
        ("Waktu run", datetime.now().isoformat(timespec='seconds')),
        ("Argumen perintah", " ".join(sys.argv[1:]) or "(tanpa argumen)"),
        ("Jumlah video di daftar run ini", n_videos),
        ("Skrip", os.path.basename(script_path)),
        ("Skrip SHA256", sha256_of(script_path)),
        ("Database wajah", FACE_DB_PATH),
        ("Database SHA256", sha256_of(FACE_DB_PATH)),
        ("Jumlah user di database", len(gallery)),
        ("User terdaftar", ", ".join(sorted(gallery.keys()))),
        ("RECOGNITION_THRESHOLD", RECOGNITION_THRESHOLD),
        ("YOLO_CONF", YOLO_CONF),
        ("YOLO_IMGSZ", YOLO_IMGSZ),
        ("YOLO_DEVICE", YOLO_DEVICE),
        ("Model deteksi", "yolov8n.pt, kelas 0 (person)"),
        ("Model wajah", "buffalo_l, CPUExecutionProvider, modul: detection+recognition"),
        ("Tracker", "bytetrack.yaml (bawaan Ultralytics)"),
        ("FACE_REC_SKIP_FRAMES (frame % N == 0)", FACE_REC_SKIP_FRAMES),
        ("FACE_DET_SIZE", str(FACE_DET_SIZE)),
        ("CROP_MODE", CROP_MODE),
        ("HEAD_FRAC / HEAD_PAD / HEAD_TOP_PAD", f"{HEAD_FRAC} / {HEAD_PAD} / {HEAD_TOP_PAD}"),
        ("MIN_FACE_SIZE (potongan, px)", MIN_FACE_SIZE),
        ("MIN_MOVE_EXTENT (px)", MIN_MOVE_EXTENT),
        ("USE_FALLBACK (identitas dari track lain)", USE_FALLBACK),
        ("Pemilihan subjek", "track bergerak yang menyeberang, extent terbesar; arah = net crossing"),
        ("Set pengembangan (hari)", ", ".join(sorted(DEV_DAYS))),
        ("LINE_START", f"({LINE_START.x}, {LINE_START.y})"),
        ("LINE_END", f"({LINE_END.x}, {LINE_END.y})"),
        ("Anchor garis", "BOTTOM_CENTER"),
        ("Titik jelajah gerak", "top-center bounding box"),
        ("Simpan video berlabel", SAVE_ANNOTATED_VIDEOS),
        ("Debug crop aktif", debug_on),
        ("Python", platform.python_version()),
        ("OS", platform.platform()),
        ("CPU (platform.processor)", platform.processor()),
        ("Thread logis (os.cpu_count)", os.cpu_count()),
        ("torch", torch_ver),
        ("torch.cuda.is_available", cuda),
    ]
    for pkg in ("ultralytics", "supervision", "insightface", "onnxruntime",
                "opencv-python", "numpy", "pandas", "openpyxl", "psutil"):
        rows.append((f"versi {pkg}", pkg_version(pkg)))
    for k, v in bytetrack_params().items():
        rows.append((f"bytetrack.yaml: {k}", v))
    return pd.DataFrame([(a, str(b)) for a, b in rows], columns=['Parameter', 'Nilai'])


def parse_args():
    p = argparse.ArgumentParser(description="Evaluasi absensi CCTV")
    p.add_argument("targets", nargs="*",
                   help="File .mp4 atau folder. Kosong = seluruh DATASET_DIR")
    p.add_argument("--save", action="store_true",
                   help="Simpan video berlabel (menimpa SAVE_ANNOTATED_VIDEOS)")
    p.add_argument("--out", default=None,
                   help="Nama file Excel hasil, misal hasil_H2_bagian1")
    p.add_argument("--resume", action="store_true",
                   help="Lanjutkan: lewati klip yang sudah ada di CSV hasil (perlu --out jika ada targets)")
    p.add_argument("--crop", choices=["body", "head"], default=None,
                   help="Mode crop wajah (bawaan: nilai CROP_MODE di skrip)")
    p.add_argument("--fallback", action="store_true",
                   help="Aktifkan fallback identitas dari track bergerak lain (analisis tambahan)")
    p.add_argument("--debug-faces", action="store_true",
                   help="Simpan crop wajah + ikhtisar track ke debug_faces/<klip>/ (waktu TIDAK valid)")
    p.add_argument("--debug-track", type=int, default=None,
                   help="Dengan --debug-faces: hanya track ID ini")
    return p.parse_args()


def collect_videos(targets):
    if not targets:
        if not os.path.exists(DATASET_DIR):
            print(f"ERROR: Folder '{DATASET_DIR}' tidak ditemukan!")
            return []
        return sorted(glob.glob(os.path.join(DATASET_DIR, "**", "*.mp4"), recursive=True))

    files = []
    for t in targets:
        if os.path.isdir(t):
            files += glob.glob(os.path.join(t, "**", "*.mp4"), recursive=True)
        elif os.path.isfile(t):
            files.append(t)
        else:
            print(f"[WARN] Tidak ditemukan: {t}")
    return sorted(set(files))


# ============================================================
# PROCESS SINGLE VIDEO CLIP
# ============================================================
def process_single_video(video_path, yolo_model, face_app, gallery, status_prefix,
                         gt_user="?", gt_action="?", debug_dir=None, debug_track=None):
    reset_tracker(yolo_model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    line_zone = sv.LineZone(start=LINE_START, end=LINE_END,
                            triggering_anchors=[sv.Position.BOTTOM_CENTER])
    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(text_scale=1.0, text_thickness=2)

    face_buffer = TrackFaceBuffer()

    out_video, out_path = None, None
    out_w = int(width * OUTPUT_SCALE) // 2 * 2
    out_h = int(height * OUTPUT_SCALE) // 2 * 2
    if SAVE_ANNOTATED_VIDEOS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        rel = re.sub(r'[\\/:]+', '_', rel_name(video_path))
        out_path = os.path.join(OUTPUT_DIR, f"result_{rel}")
        out_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
        if not out_video.isOpened():
            print(f"\n[WARN] VideoWriter gagal dibuka untuk {out_path}")
            out_video = None

    crossings = []       # {'status', 'track_id', 'frame', 'ext'}
    obs_log = []         # observasi wajah per frame
    track_max = {}       # skor mentah tertinggi per track
    track_stats = {}     # jelajah titik kepala per track
    debug_counters = {}
    frame_count = 0
    flash_text, flash_until = "", 0
    t_decode = t_yolo = t_face = 0.0
    n_face_calls = 0
    start_proc_time = time.time()
    t_loop0 = time.perf_counter()

    try:
        while True:
            t = time.perf_counter()
            ret, frame = cap.read()
            t_decode += time.perf_counter() - t
            if not ret:
                break
            frame_count += 1

            t = time.perf_counter()
            results = yolo_model.track(
                frame, classes=[0], conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=YOLO_DEVICE,
                persist=True, tracker='bytetrack.yaml', verbose=False
            )[0]
            t_yolo += time.perf_counter() - t
            detections = sv.Detections.from_ultralytics(results)

            if detections.tracker_id is not None:
                if debug_dir and frame_count % DEBUG_OVERVIEW_EVERY == 0 and len(detections) > 0:
                    save_overview(frame, detections, frame_count, debug_dir)

                do_face = (frame_count % FACE_REC_SKIP_FRAMES == 0)
                for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
                    tid = int(track_id)
                    bx1, by1, bx2, by2 = map(float, xyxy)

                    # Catat gerak (titik kepala = top-center box)
                    hx, hy = (bx1 + bx2) / 2.0, by1
                    s = track_stats.get(tid)
                    if s is None:
                        track_stats[tid] = {'minx': hx, 'maxx': hx, 'miny': hy, 'maxy': hy,
                                            'first': frame_count, 'last': frame_count, 'seen': 1}
                    else:
                        s['minx'], s['maxx'] = min(s['minx'], hx), max(s['maxx'], hx)
                        s['miny'], s['maxy'] = min(s['miny'], hy), max(s['maxy'], hy)
                        s['last'] = frame_count
                        s['seen'] += 1

                    if not do_face:
                        continue

                    if CROP_MODE == "head":
                        crop = head_crop(frame, (bx1, by1, bx2, by2))
                    else:
                        x1, y1 = max(0, int(bx1)), max(0, int(by1))
                        x2, y2 = min(width, int(bx2)), min(height, int(by2))
                        crop = frame[y1:y2, x1:x2]
                    if crop is None or crop.shape[0] <= MIN_FACE_SIZE or crop.shape[1] <= MIN_FACE_SIZE:
                        continue

                    t = time.perf_counter()
                    faces = face_app.get(crop)
                    t_face += time.perf_counter() - t
                    n_face_calls += 1
                    if len(faces) == 0:
                        continue

                    if CROP_MODE == "head":
                        face = pick_own_face(faces, crop.shape)
                    else:
                        face = max(faces, key=lambda f: f.det_score)

                    if debug_dir and (debug_track is None or tid == debug_track):
                        save_debug_face(crop, faces, face, tid, frame_count, gallery,
                                        debug_dir, debug_counters)

                    if face is None:
                        continue

                    best_user, best_sim = match_face(face, gallery)
                    track_max[tid] = max(track_max.get(tid, 0.0), best_sim)

                    # Catat SEMUA observasi (sebelum threshold) agar threshold bisa di-sweep offline
                    obs_log.append({'Track': tid, 'Frame': frame_count,
                                    'Best_User': best_user, 'Best_Score': round(best_sim, 6),
                                    'Det_Score': round(float(face.det_score), 4),
                                    'N_Faces': len(faces)})

                    if best_sim >= RECOGNITION_THRESHOLD:
                        face_buffer.add_recognition(tid, best_user, best_sim)
                    else:
                        face_buffer.add_recognition(tid, "Unknown", best_sim)

            # Line crossing (dicatat semua; subjek dipilih setelah video selesai)
            crossed_in, crossed_out = line_zone.trigger(detections)
            if detections.tracker_id is not None:
                for idx, (is_in, is_out) in enumerate(zip(crossed_in, crossed_out)):
                    if is_in or is_out:
                        status_now = "MASUK" if is_in else "KELUAR"
                        ctid = int(detections.tracker_id[idx])
                        ext_now = track_extent(track_stats[ctid]) if ctid in track_stats else 0.0
                        crossings.append({'status': status_now, 'track_id': ctid,
                                          'frame': frame_count, 'ext': ext_now})
                        flash_text = f"CROSSING: {status_now}"
                        flash_until = frame_count + int(fps)

            if frame_count > flash_until:
                flash_text = ""

            # Video anotasi
            if out_video is not None:
                annotated = frame.copy()
                if detections.tracker_id is not None and len(detections) > 0:
                    labels = []
                    for track_id in detections.tracker_id:
                        tid = int(track_id)
                        name, _, _ = face_buffer.get_final_identity(tid)
                        sc = track_max.get(tid)
                        moving_now = (tid in track_stats and
                                      track_extent(track_stats[tid]) >= MIN_MOVE_EXTENT)
                        tag = " [M]" if moving_now else ""
                        if sc is None:
                            labels.append(f"ID:{tid} mencari wajah...{tag}")
                        else:
                            labels.append(f"ID:{tid} {name} ({sc:.2f}){tag}")
                    annotated = box_annotator.annotate(scene=annotated, detections=detections)
                    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

                cv2.line(annotated, (LINE_START.x, LINE_START.y), (LINE_END.x, LINE_END.y), (0, 255, 255), 4)

                hud = (f"Frame {frame_count}/{total_frames} | GT: {gt_user}-{gt_action} | "
                       f"IN:{line_zone.in_count} OUT:{line_zone.out_count}")
                draw_hud(annotated, hud, HUD_POSITION, scale=HUD_SCALE)
                if flash_text:
                    draw_hud(annotated, flash_text, HUD_POSITION, scale=HUD_SCALE * 1.2,
                             color=(80, 80, 255), y_offset=-60 if "bottom" in HUD_POSITION else 60)

                if OUTPUT_SCALE != 1.0:
                    annotated = cv2.resize(annotated, (out_w, out_h))
                out_video.write(annotated)

            if frame_count % 5 == 0 or frame_count == total_frames:
                elapsed = time.time() - start_proc_time
                curr_fps = frame_count / elapsed if elapsed > 0 else 0
                pct = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"\r{status_prefix} | Frame: {frame_count}/{total_frames} ({pct:.1f}%) | {curr_fps:.1f} FPS",
                      end="", flush=True)
    finally:
        cap.release()
        if out_video is not None:
            out_video.release()

    loop_time = time.perf_counter() - t_loop0

    # ---------------- Pemilihan subjek (setelah video selesai) ----------------
    moving_tids = {tid for tid, s in track_stats.items() if track_extent(s) >= MIN_MOVE_EXTENT}

    by_track, crossing_count = {}, {}
    for c in crossings:
        by_track.setdefault(c['track_id'], []).append(c)
        crossing_count[c['track_id']] = crossing_count.get(c['track_id'], 0) + 1

    candidates = [tid for tid in by_track if tid in moving_tids]

    obs_count = {}
    for o in obs_log:
        obs_count[o['Track']] = obs_count.get(o['Track'], 0) + 1

    events = []
    subject_tid, subject_ext = None, 0.0
    if candidates:
        subject_tid = max(candidates, key=lambda t: track_extent(track_stats[t]))
        subject_ext = track_extent(track_stats[subject_tid])
        evs = by_track[subject_tid]
        net = sum(1 if e['status'] == "MASUK" else -1 for e in evs)
        if net > 0:
            status = "MASUK"
        elif net < 0:
            status = "KELUAR"
        else:
            status = evs[-1]['status']
        name, conf, n, source = resolve_identity(face_buffer, subject_tid, moving_tids)
        events.append({'status': status, 'name': name, 'confidence': conf,
                       'frames_count': n, 'source': source, 'track_id': subject_tid})

    track_rows = []
    for tid in sorted(track_stats):
        s = track_stats[tid]
        track_rows.append({
            'Track': tid, 'First_Frame': s['first'], 'Last_Frame': s['last'],
            'Frames_Seen': s['seen'],
            'MinX': round(s['minx'], 2), 'MaxX': round(s['maxx'], 2),
            'MinY': round(s['miny'], 2), 'MaxY': round(s['maxy'], 2),
            'Extent_px': round(track_extent(s), 2),
            'Face_Obs': obs_count.get(tid, 0),
            'Crossings': crossing_count.get(tid, 0),
        })
    cross_rows = [{'Track': c['track_id'], 'Frame': c['frame'], 'Arah': c['status'],
                   'Extent_at_Crossing_px': round(c['ext'], 2)} for c in crossings]

    info = {
        'events': events,
        'max_raw': max(track_max.values()) if track_max else 0.0,
        'subject_tid': subject_tid,
        'subject_ext': subject_ext,
        'subject_max': track_max.get(subject_tid, 0.0) if subject_tid is not None else 0.0,
        'subject_face_obs': obs_count.get(subject_tid, 0) if subject_tid is not None else 0,
        'event_lain': max(0, len(candidates) - 1),
        'raw_crossings': len(crossings),
        'moving_tracks': len(moving_tids),
        'track_summary': build_track_summary(track_stats, face_buffer, track_max,
                                             crossing_count, moving_tids),
        'out_path': out_path,
        'frames_read': frame_count,
        'video_fps': fps,
        'obs_log': obs_log,
        'cross_rows': cross_rows,
        'track_rows': track_rows,
        'timing': {'decode': t_decode, 'yolo': t_yolo, 'face': t_face,
                   'other': max(0.0, loop_time - t_decode - t_yolo - t_face),
                   'face_calls': n_face_calls},
    }
    return info


# ============================================================
# REKAP
# ============================================================
def group_table(df, keys):
    tmp = df.assign(_ok=(df['Kategori'] == 'benar'))
    t = tmp.groupby(keys).agg(Klip=('_ok', 'size'), Benar=('_ok', 'sum')).reset_index()
    t['Benar'] = t['Benar'].astype(int)
    t['Akurasi_%'] = (t['Benar'] / t['Klip'] * 100).round(1)
    cis = [wilson(int(k), int(n)) for k, n in zip(t['Benar'], t['Klip'])]
    t['CI95_bawah_%'] = [round(c[0], 1) for c in cis]
    t['CI95_atas_%'] = [round(c[1], 1) for c in cis]
    return t


CAT_ORDER = ['benar', 'wrong_direction', 'unknown', 'no_crossing', 'wrong_identity']


# ============================================================
# MAIN BATCH EVALUATION
# ============================================================
def main():
    global SAVE_ANNOTATED_VIDEOS, OUTPUT_EXCEL, CROP_MODE, USE_FALLBACK
    args = parse_args()

    if args.save:
        SAVE_ANNOTATED_VIDEOS = True
    if args.crop:
        CROP_MODE = args.crop
    if args.fallback:
        USE_FALLBACK = True
    debug_on = bool(args.debug_faces)

    if args.targets and args.resume and not args.out:
        print("ERROR: --resume dengan targets memerlukan --out agar nama berkas stabil.")
        return

    # Uji parsial: nama berstempel waktu agar tidak menimpa hasil lain
    if args.targets:
        OUTPUT_EXCEL = f"hasil_evaluasi_parsial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    if args.out:
        OUTPUT_EXCEL = args.out if args.out.endswith(".xlsx") else args.out + ".xlsx"
    paths = out_paths(OUTPUT_EXCEL)

    # Tanpa --resume, berkas lama dengan nama sama disingkirkan (tidak ditimpa/tercampur)
    if not args.resume:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for p in paths.values():
            if os.path.exists(p):
                os.replace(p, f"{p}.old_{stamp}")

    video_files = collect_videos(args.targets)
    if not video_files:
        print("Tidak ada video untuk diproses.")
        return

    print("1. Memuat Database Wajah & Model AI...")
    with open(FACE_DB_PATH, 'rb') as f:
        raw_db = pickle.load(f)
    gallery = build_gallery(raw_db)
    print(f"   Database: {len(gallery)} user, {sum(len(v) for v in gallery.values())} embedding")
    print(f"   User terdaftar: {sorted(gallery.keys())}")

    yolo_model = YOLO('yolov8n.pt')
    face_app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'],
                            providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1, det_size=FACE_DET_SIZE)
    print("   Semua model berhasil dimuat!\n")

    done = done_clips(paths['hasil']) if args.resume else set()
    if done:
        print(f"[RESUME] {len(done)} klip sudah selesai, dilewati.")

    print("=" * 110)
    print(f"MEMULAI UJI: {len(video_files)} VIDEO | Threshold = {RECOGNITION_THRESHOLD} "
          f"| MinMove = {MIN_MOVE_EXTENT}px | Crop = {CROP_MODE} | Fallback = {USE_FALLBACK} "
          f"| Device = {YOLO_DEVICE}")
    print(f"Output Excel: {OUTPUT_EXCEL}")
    if SAVE_ANNOTATED_VIDEOS or debug_on:
        print("[CATATAN] --save / --debug-faces aktif: waktu proses TIDAK valid untuk data RQ3.")
    print("=" * 110)

    session_count = 0

    for idx, video_path in enumerate(video_files, 1):
        rel_path = rel_name(video_path)
        if rel_path in done:
            continue
        gt_user, gt_action, gt_day = parse_ground_truth(rel_path)
        status_prefix = f"[{idx}/{len(video_files)}] {rel_path}"

        if gt_user == "UNKNOWN" or gt_action == "UNKNOWN":
            print(f"[SKIP] Ground truth tidak lengkap untuk '{rel_path}' "
                  f"(user={gt_user}, aksi={gt_action}).")
            append_df(paths['err'], pd.DataFrame([{'Path_Video': rel_path,
                                                   'Error': 'GT tidak lengkap, dilewati'}]))
            continue

        debug_dir = None
        if debug_on:
            tag = re.sub(r'[\\/:]+', '_', os.path.splitext(rel_path)[0])
            debug_dir = os.path.join(DEBUG_DIR_ROOT, f"{tag}_{CROP_MODE}")

        try:
            t0 = time.perf_counter()
            info = process_single_video(video_path, yolo_model, face_app, gallery, status_prefix,
                                        gt_user=gt_user, gt_action=gt_action,
                                        debug_dir=debug_dir, debug_track=args.debug_track)
            elapsed_vid = time.perf_counter() - t0
        except Exception as e:
            print(f"\n[ERROR] {rel_path}: {e!r}")
            append_df(paths['err'], pd.DataFrame([{'Path_Video': rel_path, 'Error': repr(e)}]))
            continue

        events, max_raw, out_path = info['events'], info['max_raw'], info['out_path']

        if len(events) == 0:
            pred_user, pred_action = "NO_DETECTION", "NO_CROSSING"
            conf, source, votes = 0.0, "NONE", 0
        else:
            ev = events[0]
            pred_user, pred_action = ev['name'], ev['status']
            conf, source, votes = ev['confidence'], ev['source'], ev['frames_count']
            if pred_user == "Unknown":
                votes = 0

        kategori = classify_clip(gt_user, gt_action, pred_user, pred_action)
        is_user_correct = (pred_user == gt_user)
        is_action_correct = (pred_action == gt_action)
        is_overall_correct = is_user_correct and is_action_correct
        status_icon = "[OK]      " if is_overall_correct else "[MISMATCH]"

        # Akhiran _OK / _MISMATCH pada video anotasi
        if out_path and os.path.exists(out_path):
            base, ext = os.path.splitext(out_path)
            new_path = f"{base}_{'OK' if is_overall_correct else 'MISMATCH'}{ext}"
            if os.path.exists(new_path):
                os.remove(new_path)
            os.replace(out_path, new_path)
            out_path = new_path

        final_text = (f"\r{status_prefix} | {status_icon} ({elapsed_vid:.1f}s) | "
                      f"GT: {gt_user}-{gt_action} | Pred: {pred_user}-{pred_action} | "
                      f"Conf: {conf:.2f} | MaxRaw: {max_raw:.2f} | Src: {source} | {kategori}")
        print(final_text.ljust(150))
        if not is_overall_correct:
            print(f"     raw_crossing={info['raw_crossings']} | track_bergerak={info['moving_tracks']} "
                  f"| subjek=T{info['subject_tid']}")
            print(f"     {info['track_summary']}")

        # Tulis per klip (aman dari listrik mati / crash)
        append_df(paths['obs'], pd.DataFrame(
            [{'Path_Video': rel_path, **o} for o in info['obs_log']],
            columns=['Path_Video', 'Track', 'Frame', 'Best_User', 'Best_Score', 'Det_Score', 'N_Faces']))
        append_df(paths['cross'], pd.DataFrame(
            [{'Path_Video': rel_path, **c} for c in info['cross_rows']],
            columns=['Path_Video', 'Track', 'Frame', 'Arah', 'Extent_at_Crossing_px']))
        append_df(paths['track'], pd.DataFrame(
            [{'Path_Video': rel_path, **t} for t in info['track_rows']],
            columns=['Path_Video', 'Track', 'First_Frame', 'Last_Frame', 'Frames_Seen', 'MinX', 'MaxX',
                     'MinY', 'MaxY', 'Extent_px', 'Face_Obs', 'Crossings']))

        duration_s = info['frames_read'] / info['video_fps'] if info['video_fps'] > 0 else 0.0
        tm = info['timing']
        margin = round(conf - RECOGNITION_THRESHOLD, 4) if votes > 0 else None
        row = {
            'Path_Video': rel_path,
            'Set': 'PENGEMBANGAN' if gt_day in DEV_DAYS else 'UJI',
            'GT_Day': gt_day,
            'GT_User': gt_user,
            'GT_Registered': gt_user in gallery,
            'GT_Action': gt_action,
            'Pred_User': pred_user,
            'Pred_Action': pred_action,
            'Kategori': kategori,
            'Confidence': round(conf, 4),
            'Max_Raw_Score': round(max_raw, 4),
            'Subject_Max_Score': round(info['subject_max'], 4),
            'Subject_Track': info['subject_tid'],
            'Subject_Extent_px': round(info['subject_ext'], 1),
            'Subject_Votes': votes,
            'Subject_Face_Obs': info['subject_face_obs'],
            'Margin_Threshold': margin,
            'Identity_Source': source,
            'Event_Lain': info['event_lain'],
            'Num_Raw_Crossings': info['raw_crossings'],
            'Num_Moving_Tracks': info['moving_tracks'],
            'User_Correct': is_user_correct,
            'Action_Correct': is_action_correct,
            'Overall_Correct': is_overall_correct,
            'Crop_Mode': CROP_MODE,
            'Frames': info['frames_read'],
            'Video_FPS': round(info['video_fps'], 2),
            'Durasi_Video_s': round(duration_s, 1),
            'Waktu_Proses_s': round(elapsed_vid, 2),
            'FPS_Proses': round(info['frames_read'] / elapsed_vid, 2) if elapsed_vid > 0 else None,
            'RTF': round(elapsed_vid / duration_s, 2) if duration_s > 0 else None,
            'T_Decode_s': round(tm['decode'], 2),
            'T_YOLO_s': round(tm['yolo'], 2),
            'T_Face_s': round(tm['face'], 2),
            'T_Lain_s': round(tm['other'], 2),
            'N_Face_Calls': tm['face_calls'],
            'RAM_Puncak_MB': peak_ram_mb(),
            'Is_Warmup': session_count == 0,
            'Valid_Waktu': not (SAVE_ANNOTATED_VIDEOS or debug_on),
            'Track_Summary': info['track_summary'],
            'Video_Output': out_path or "",
        }
        append_df(paths['hasil'], pd.DataFrame([row]))
        session_count += 1

    # ---------------- REKAPITULASI (dari CSV, mencakup klip hasil resume) ----------------
    if not os.path.exists(paths['hasil']):
        print("\nTidak ada klip yang berhasil diproses.")
        return

    df = pd.read_csv(paths['hasil'], encoding="utf-8-sig")
    n = len(df)
    for c in ('User_Correct', 'Action_Correct', 'Overall_Correct', 'GT_Registered',
              'Is_Warmup', 'Valid_Waktu'):
        df[c] = df[c].astype(bool)

    acc_user = df['User_Correct'].sum() / n * 100
    acc_action = df['Action_Correct'].sum() / n * 100
    acc_overall = df['Overall_Correct'].sum() / n * 100
    lo, hi = wilson(int(df['Overall_Correct'].sum()), n)

    claimed = df[~df['Pred_User'].isin(["Unknown", "NO_DETECTION"])]
    n_false_accept = int((~claimed['User_Correct']).sum())
    n_unknown = int((df['Pred_User'] == "Unknown").sum())
    n_no_cross = int((df['Pred_Action'] == "NO_CROSSING").sum())
    precision_user = (claimed['User_Correct'].sum() / len(claimed) * 100) if len(claimed) else 0.0

    reg = df[df['GT_Registered']]
    n_unreg = n - len(reg)

    tdf = df[df['Valid_Waktu'] & ~df['Is_Warmup']]
    total_time = float(df['Waktu_Proses_s'].sum())
    mean_t = float(tdf['Waktu_Proses_s'].mean()) if len(tdf) else float('nan')
    median_t = float(tdf['Waktu_Proses_s'].median()) if len(tdf) else float('nan')
    sd_t = float(tdf['Waktu_Proses_s'].std()) if len(tdf) > 1 else float('nan')
    mean_rtf = float(tdf['RTF'].mean()) if len(tdf) else float('nan')

    print("\n" + "=" * 110)
    print("HASIL EVALUASI")
    print("=" * 110)
    print(f"Total klip                 : {n}  (GT tidak terdaftar: {n_unreg})")
    print(f"Waktu pemrosesan (jumlah)  : {total_time:.1f} detik ({total_time/60:.1f} menit)")
    print(f"Waktu per klip (valid)     : rata-rata {mean_t:.1f} s | median {median_t:.1f} s | SD {sd_t:.1f} s "
          f"| RTF {mean_rtf:.2f}  [n={len(tdf)}, warm-up & --save/--debug dikecualikan]")
    print(f"Akurasi identitas          : {acc_user:.2f}% ({df['User_Correct'].sum()}/{n})")
    print(f"Precision identitas        : {precision_user:.2f}% ({claimed['User_Correct'].sum()}/{len(claimed)})")
    print(f"Salah identitas (FA)       : {n_false_accept}/{n}")
    print(f"Unknown (FR)               : {n_unknown}/{n}")
    print(f"Tanpa crossing             : {n_no_cross}/{n}")
    print(f"Akurasi line crossing      : {acc_action:.2f}% ({df['Action_Correct'].sum()}/{n})")
    print(f"AKURASI END-TO-END         : {acc_overall:.2f}% ({df['Overall_Correct'].sum()}/{n})  "
          f"CI95 Wilson [{lo:.1f}-{hi:.1f}]")
    print("-" * 110)
    for s_name, sub in df.groupby('Set'):
        k = int((sub['Kategori'] == 'benar').sum())
        l2, h2 = wilson(k, len(sub))
        cats = ", ".join(f"{c}={int((sub['Kategori'] == c).sum())}" for c in CAT_ORDER)
        print(f"Set {s_name:<13}: {k}/{len(sub)} benar ({k/len(sub)*100:.1f}%, CI95 {l2:.1f}-{h2:.1f}) | {cats}")
    print("=" * 110)

    summary_rows = [
        ('Total klip', n),
        ('GT tidak terdaftar', n_unreg),
        ('Akurasi identitas (%)', round(acc_user, 2)),
        ('Precision identitas (%)', round(precision_user, 2)),
        ('Salah identitas / FA (jumlah)', n_false_accept),
        ('Unknown / FR (jumlah)', n_unknown),
        ('Tanpa crossing (jumlah)', n_no_cross),
        ('Akurasi line crossing (%)', round(acc_action, 2)),
        ('Akurasi end-to-end (%)', round(acc_overall, 2)),
        ('CI95 Wilson end-to-end bawah (%)', round(lo, 1)),
        ('CI95 Wilson end-to-end atas (%)', round(hi, 1)),
        ('Threshold', RECOGNITION_THRESHOLD),
        ('MIN_MOVE_EXTENT (px)', MIN_MOVE_EXTENT),
        ('Crop mode', CROP_MODE),
        ('Fallback', USE_FALLBACK),
        ('Waktu pemrosesan jumlah (detik)', round(total_time, 1)),
        ('Waktu rata-rata per klip, valid (detik)', round(mean_t, 2)),
        ('Waktu median per klip, valid (detik)', round(median_t, 2)),
        ('Waktu SD per klip, valid (detik)', round(sd_t, 2)),
        ('RTF rata-rata, valid', round(mean_rtf, 2)),
        ('Jumlah klip untuk statistik waktu', len(tdf)),
    ]
    summary = pd.DataFrame(summary_rows, columns=['Metrik', 'Nilai'])

    kat_tab = pd.crosstab(df['Set'], df['Kategori'], margins=True)
    per_set = group_table(df, ['Set'])
    per_hari = group_table(df, ['Set', 'GT_Day'])
    per_user = group_table(df, ['GT_User'])
    cm_action = pd.crosstab(df['GT_Action'], df['Pred_Action'])
    cm_user = pd.crosstab(df['GT_User'], df['Pred_User'])

    def read_opt(p):
        return pd.read_csv(p, encoding="utf-8-sig") if os.path.exists(p) else pd.DataFrame()

    err_df = read_opt(paths['err'])
    config_df = build_config_table(gallery, len(video_files), debug_on)

    sheets = {
        'Hasil': (df, False),
        'Ringkasan': (summary, False),
        'Kategori': (kat_tab, True),
        'Per_Set': (per_set, False),
        'Per_Hari': (per_hari, False),
        'Per_Karyawan': (per_user, False),
        'Konfigurasi': (config_df, False),
        'CM_Aksi': (cm_action, True),
        'CM_Identitas': (cm_user, True),
        'Track': (read_opt(paths['track']), False),
        'Crossing': (read_opt(paths['cross']), False),
        'Observasi_Wajah': (read_opt(paths['obs']), False),
    }
    if len(err_df):
        sheets['Error'] = (err_df, False)
    saved_path = save_excel_safe(sheets, OUTPUT_EXCEL)

    print(f"\nHasil rekap disimpan ke: '{saved_path}'")
    if len(err_df):
        print(f"[WARN] {len(err_df)} klip gagal/dilewati, lihat '{paths['err']}'")
    if SAVE_ANNOTATED_VIDEOS:
        print(f"Video anotasi disimpan di folder: '{OUTPUT_DIR}/'")
    if debug_on:
        print(f"Gambar debug di folder: '{DEBUG_DIR_ROOT}/'")


if __name__ == "__main__":
    main()