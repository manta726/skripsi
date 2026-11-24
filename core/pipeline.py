"""Pipeline absensi CCTV: YOLOv8n + ByteTrack + InsightFace + line crossing.
Logika disalin dari run_batch_evaluation.py (konfigurasi root).
Parameter adalah variabel modul; halaman Streamlit mengisinya sebelum proses."""
import math
import os
import re
import subprocess
import time

import cv2
from core import waktu as W
import numpy as np
import supervision as sv

# ============================================================
# PARAMETER (nilai awal = konfigurasi run_batch_evaluation.py root)
# ============================================================
RECOGNITION_THRESHOLD = 0.35
YOLO_CONF = 0.3
YOLO_IMGSZ = 1280
YOLO_DEVICE = "cpu"
FACE_REC_SKIP_FRAMES = 3
FACE_DET_SIZE = (640, 640)
MIN_FACE_SIZE = 30
MIN_MOVE_EXTENT = 150
ATURAN_ARAH = "net"        # "net" | "pertama" | "terakhir"
TOLERANSI_WAKTU_S = 2.0
CROP_MODE = "body"            # "body" | "head"
HEAD_FRAC = 0.45
HEAD_PAD = 0.10
HEAD_TOP_PAD = 0.05
USE_FALLBACK = False
SAVE_ANNOTATED_VIDEOS = True
OUTPUT_MAX_WIDTH = 1152       # lebar maksimum video keluaran
HUD_POSITION = "bottom-left"
HUD_SCALE = 0.9
REF_WIDTH = 2304              # resolusi acuan untuk menskalakan font/garis
LINE_START = (1554, 610)
LINE_END = (408, 1146)
DEV_DAYS = {"H1", "H2"}


# ============================================================
# HELPER
# ============================================================
def parse_ground_truth(name):
    """Ekstrak GT user, aksi, hari dari nama file / path."""
    path_clean = name.replace("\\", "/")
    m = re.search(r'(?<![A-Za-z])K\d{3}(?!\d)', path_clean, re.IGNORECASE)
    gt_user = m.group(0).upper() if m else "UNKNOWN"

    stem = os.path.splitext(os.path.basename(path_clean))[0].upper()
    tokens = [t for t in re.split(r'[_\-\s.]+', stem) if t]
    up = path_clean.upper()
    if any(t in ("IN", "MASUK") for t in tokens) or "/IN/" in up or "/MASUK/" in up:
        gt_action = "MASUK"
    elif any(t in ("OUT", "KELUAR") for t in tokens) or "/OUT/" in up or "/KELUAR/" in up:
        gt_action = "KELUAR"
    else:
        gt_action = "UNKNOWN"

    d = re.search(r'(?<![A-Za-z])(H\d+|Hari[_\s]?\d+)', path_clean, re.IGNORECASE)
    gt_day = d.group(1).upper() if d else "H?"
    return gt_user, gt_action, gt_day


def reset_tracker(yolo_model):
    """Bersihkan state ByteTrack agar tidak bocor antar video."""
    try:
        if hasattr(yolo_model, "predictor") and yolo_model.predictor is not None:
            trackers = getattr(yolo_model.predictor, "trackers", None)
            if trackers:
                for t in trackers:
                    t.reset()
            yolo_model.predictor = None
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
        self.buffer.setdefault(track_id, []).append({"name": name, "score": score})

    def get_final_identity(self, track_id):
        if track_id not in self.buffer or len(self.buffer[track_id]) == 0:
            return "Unknown", 0.0, 0
        votes = self.buffer[track_id]
        user_scores, user_counts = {}, {}
        for item in votes:
            if item["name"] != "Unknown":
                user_scores[item["name"]] = user_scores.get(item["name"], 0.0) + item["score"]
                user_counts[item["name"]] = user_counts.get(item["name"], 0) + 1
        if not user_scores:
            return "Unknown", 0.0, len(votes)
        best = max(user_scores, key=user_scores.get)
        return best, user_scores[best] / user_counts[best], user_counts[best]


def track_extent(stats):
    return math.hypot(stats["maxx"] - stats["minx"], stats["maxy"] - stats["miny"])


def head_crop(frame, xyxy, frac=None, pad=None, top_pad=None):
    frac = HEAD_FRAC if frac is None else frac
    pad = HEAD_PAD if pad is None else pad
    top_pad = HEAD_TOP_PAD if top_pad is None else top_pad
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


def resolve_identity(face_buffer, track_id, moving_tids):
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
        peak = getattr(mi, "peak_wset", None)
        return round((peak if peak else mi.rss) / 1048576, 1)
    except Exception:
        return None


def draw_hud(img, text, position="bottom-left", scale=0.9, thickness=2,
             color=(0, 255, 255), alpha=0.55, margin=15, y_offset=0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    pad = max(4, int(10 * scale / 0.9))
    box_w, box_h = tw + 2 * pad, th + base + 2 * pad
    x0 = margin if "left" in position else w - box_w - margin
    y0 = margin if "top" in position else h - box_h - margin
    y0 += y_offset
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x0 + box_w), min(h, y0 + box_h)
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0, roi)
    cv2.putText(img, text, (x0 + pad, y0 + pad + th), font, scale, color, thickness, cv2.LINE_AA)


# ============================================================
# PROSES SATU KLIP
# ============================================================
def process_single_video(video_path, yolo_model, face_app, gallery,
                         gt_user="?", gt_action="?",
                         raw_out_path=None, progress_cb=None):
    t_func0 = time.perf_counter()
    reset_tracker(yolo_model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_fallback = False
    if fps <= 0:
        fps = 20.0
        fps_fallback = True
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    p_start = sv.Point(int(LINE_START[0]), int(LINE_START[1]))
    p_end = sv.Point(int(LINE_END[0]), int(LINE_END[1]))
    line_zone = sv.LineZone(start=p_start, end=p_end,
                            triggering_anchors=[sv.Position.BOTTOM_CENTER])

    # skala anotasi terhadap resolusi
    k = max(0.6, width / REF_WIDTH)
    thick = max(1, int(round(4 * k)))
    box_annotator = sv.BoxAnnotator(thickness=thick)
    label_annotator = sv.LabelAnnotator(text_scale=1.0 * k,
                                        text_thickness=max(1, int(round(2 * k))))

    face_buffer = TrackFaceBuffer()

    out_video = None
    out_scale = min(1.0, OUTPUT_MAX_WIDTH / max(width, 1))
    out_w = max(2, int(width * out_scale) // 2 * 2)
    out_h = max(2, int(height * out_scale) // 2 * 2)
    if raw_out_path is not None:
        out_video = cv2.VideoWriter(str(raw_out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                    fps, (out_w, out_h))
        if not out_video.isOpened():
            out_video = None

    crossings, obs_log, det_log, dets_per_frame = [], [], [], []
    track_max, track_stats = {}, {}
    frame_count = 0
    flash_text, flash_until = "", 0
    t_decode = t_yolo = t_face = t_annot = 0.0
    n_face_calls = 0
    t_wall0 = time.time()
    t_loop0 = time.perf_counter()

    try:
        while True:
            t = time.perf_counter()
            ret, frame = cap.read()
            t_decode += time.perf_counter() - t
            if not ret:
                break
            frame_count += 1

            # --- YOLOv8n + ByteTrack (satu panggilan) ---
            t = time.perf_counter()
            results = yolo_model.track(
                frame, classes=[0], conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=YOLO_DEVICE,
                persist=True, tracker="bytetrack.yaml", verbose=False
            )[0]
            t_yolo += time.perf_counter() - t
            detections = sv.Detections.from_ultralytics(results)
            dets_per_frame.append(len(detections))

            if detections.tracker_id is not None:
                confs = (detections.confidence if detections.confidence is not None
                         else np.zeros(len(detections)))
                do_face = (frame_count % FACE_REC_SKIP_FRAMES == 0)
                for xyxy, conf_d, track_id in zip(detections.xyxy, confs, detections.tracker_id):
                    tid = int(track_id)
                    bx1, by1, bx2, by2 = map(float, xyxy)

                    det_log.append({"Frame": frame_count, "Track": tid,
                                    "Conf": round(float(conf_d), 4),
                                    "X1": round(bx1, 1), "Y1": round(by1, 1),
                                    "X2": round(bx2, 1), "Y2": round(by2, 1)})

                    hx, hy = (bx1 + bx2) / 2.0, by1
                    s = track_stats.get(tid)
                    if s is None:
                        track_stats[tid] = {"minx": hx, "maxx": hx, "miny": hy, "maxy": hy,
                                            "first": frame_count, "last": frame_count, "seen": 1}
                    else:
                        s["minx"], s["maxx"] = min(s["minx"], hx), max(s["maxx"], hx)
                        s["miny"], s["maxy"] = min(s["miny"], hy), max(s["maxy"], hy)
                        s["last"] = frame_count
                        s["seen"] += 1

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

                    # --- InsightFace ---
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
                    if face is None:
                        continue

                    best_user, best_sim = match_face(face, gallery)
                    track_max[tid] = max(track_max.get(tid, 0.0), best_sim)
                    obs_log.append({"Track": tid, "Frame": frame_count,
                                    "Best_User": best_user, "Best_Score": round(best_sim, 6),
                                    "Det_Score": round(float(face.det_score), 4),
                                    "N_Faces": len(faces)})
                    if best_sim >= RECOGNITION_THRESHOLD:
                        face_buffer.add_recognition(tid, best_user, best_sim)
                    else:
                        face_buffer.add_recognition(tid, "Unknown", best_sim)

            # --- Line crossing ---
            if detections.tracker_id is not None:
                crossed_in, crossed_out = line_zone.trigger(detections)
                for idx, (is_in, is_out) in enumerate(zip(crossed_in, crossed_out)):
                    if is_in or is_out:
                        status_now = "MASUK" if is_in else "KELUAR"
                        ctid = int(detections.tracker_id[idx])
                        ext_now = track_extent(track_stats[ctid]) if ctid in track_stats else 0.0
                        crossings.append({"status": status_now, "track_id": ctid,
                                          "frame": frame_count, "ext": ext_now})
                        flash_text = f"CROSSING: {status_now}"
                        flash_until = frame_count + int(fps)
            if frame_count > flash_until:
                flash_text = ""

            # --- Anotasi video ---
            if out_video is not None:
                t = time.perf_counter()
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
                    annotated = label_annotator.annotate(scene=annotated, detections=detections,
                                                         labels=labels)
                cv2.line(annotated, (p_start.x, p_start.y), (p_end.x, p_end.y),
                         (0, 255, 255), thick)
                hud = (f"Frame {frame_count}/{total_frames} | GT: {gt_user}-{gt_action} | "
                       f"IN:{line_zone.in_count} OUT:{line_zone.out_count}")
                hs = HUD_SCALE * k
                ht = max(1, int(round(2 * k)))
                draw_hud(annotated, hud, HUD_POSITION, scale=hs, thickness=ht,
                         margin=int(15 * k))
                if flash_text:
                    draw_hud(annotated, flash_text, HUD_POSITION, scale=hs * 1.2, thickness=ht,
                             color=(80, 80, 255), margin=int(15 * k),
                             y_offset=int(-60 * k) if "bottom" in HUD_POSITION else int(60 * k))
                if out_scale != 1.0:
                    annotated = cv2.resize(annotated, (out_w, out_h))
                out_video.write(annotated)
                t_annot += time.perf_counter() - t

            if progress_cb is not None and (frame_count % 5 == 0 or frame_count == total_frames):
                el = time.time() - t_wall0
                progress_cb(frame_count, total_frames, frame_count / el if el > 0 else 0.0)
    finally:
        cap.release()
        if out_video is not None:
            out_video.release()

    loop_time = time.perf_counter() - t_loop0

    # ---------------- Pemilihan subjek ----------------
    moving_tids = {tid for tid, s in track_stats.items() if track_extent(s) >= MIN_MOVE_EXTENT}
    by_track, crossing_count = {}, {}
    for c in crossings:
        by_track.setdefault(c["track_id"], []).append(c)
        crossing_count[c["track_id"]] = crossing_count.get(c["track_id"], 0) + 1
    candidates = [tid for tid in by_track if tid in moving_tids]

    obs_count = {}
    for o in obs_log:
        obs_count[o["Track"]] = obs_count.get(o["Track"], 0) + 1

    events = []
    subject_tid, subject_ext = None, 0.0
    urutan, frame_penentu = "", None
    if candidates:
        subject_tid = max(candidates, key=lambda t_: track_extent(track_stats[t_]))
        subject_ext = track_extent(track_stats[subject_tid])
        evs = by_track[subject_tid]
        if ATURAN_ARAH == "pertama":
            chosen = evs[0]
        elif ATURAN_ARAH == "terakhir":
            chosen = evs[-1]
        else:  # "net": sama dengan perilaku sebelumnya
            net = sum(1 if e["status"] == "MASUK" else -1 for e in evs)
            if net > 0:
                status_net = "MASUK"
            elif net < 0:
                status_net = "KELUAR"
            else:
                status_net = evs[-1]["status"]
            chosen = [e for e in evs if e["status"] == status_net][-1]
        status = chosen["status"]
        frame_penentu = chosen["frame"]
        urutan = " -> ".join(f"{e['status']}@{e['frame']}" for e in evs)
        name, conf, n, source = resolve_identity(face_buffer, subject_tid, moving_tids)
        events.append({"status": status, "name": name, "confidence": conf,
                       "frames_count": n, "source": source, "track_id": subject_tid})

    track_rows = []
    for tid in sorted(track_stats):
        s = track_stats[tid]
        name, avg, n = face_buffer.get_final_identity(tid)
        track_rows.append({
            "Track": tid, "First_Frame": s["first"], "Last_Frame": s["last"],
            "Frames_Seen": s["seen"],
            "MinX": round(s["minx"], 2), "MaxX": round(s["maxx"], 2),
            "MinY": round(s["miny"], 2), "MaxY": round(s["maxy"], 2),
            "Extent_px": round(track_extent(s), 2),
            "Gerak": tid in moving_tids,
            "Face_Obs": obs_count.get(tid, 0),
            "Crossings": crossing_count.get(tid, 0),
            "Identitas_Akhir": name, "Skor_Rata2": round(avg, 4), "Votes": n,
        })
    cross_rows = [{"Track": c["track_id"], "Frame": c["frame"], "Arah": c["status"],
                   "Extent_at_Crossing_px": round(c["ext"], 2)} for c in crossings]

    elapsed = time.perf_counter() - t_func0
    return {
        "events": events,
        "max_raw": max(track_max.values()) if track_max else 0.0,
        "subject_tid": subject_tid,
        "subject_ext": subject_ext,
        "subject_max": track_max.get(subject_tid, 0.0) if subject_tid is not None else 0.0,
        "subject_face_obs": obs_count.get(subject_tid, 0) if subject_tid is not None else 0,
        "event_lain": max(0, len(candidates) - 1),
        "raw_crossings": len(crossings),
        "moving_tracks": len(moving_tids),
        "track_summary": build_track_summary(track_stats, face_buffer, track_max,
                                             crossing_count, moving_tids),
        "frames_read": frame_count,
        "total_frames": total_frames,
        "video_fps": fps,
        "fps_fallback": fps_fallback,
        "urutan_crossing": urutan,
        "frame_penentu": frame_penentu,
        "video_size": (width, height),
        "obs_log": obs_log,
        "det_log": det_log,
        "dets_per_frame": dets_per_frame,
        "cross_rows": cross_rows,
        "track_rows": track_rows,
        "elapsed": elapsed,
        "timing": {"decode": t_decode, "yolo": t_yolo, "face": t_face, "annot": t_annot,
                   "other": max(0.0, loop_time - t_decode - t_yolo - t_face - t_annot),
                   "face_calls": n_face_calls},
    }


def build_row(info, rel_path, gt_user, gt_action, gt_day, gallery, is_warmup,
              jam_mulai="", jam_aktual=""):
    events = info["events"]
    if len(events) == 0:
        pred_user, pred_action = "NO_DETECTION", "NO_CROSSING"
        conf, source, votes = 0.0, "NONE", 0
    else:
        ev = events[0]
        pred_user, pred_action = ev["name"], ev["status"]
        conf, source, votes = ev["confidence"], ev["source"], ev["frames_count"]
        if pred_user == "Unknown":
            votes = 0

    kategori = classify_clip(gt_user, gt_action, pred_user, pred_action)
    user_ok = pred_user == gt_user
    act_ok = pred_action == gt_action
    elapsed = info["elapsed"]
    duration = info["frames_read"] / info["video_fps"] if info["video_fps"] > 0 else 0.0
    tm = info["timing"]
    margin = round(conf - RECOGNITION_THRESHOLD, 4) if votes > 0 else None
    mulai = W.parse_waktu(jam_mulai)
    aktual = W.parse_waktu(jam_aktual)
    fp = info.get("frame_penentu")
    dt = W.jam_crossing(mulai, fp, info["video_fps"], 1) if fp is not None else None
    sel = W.selisih_detik(dt, aktual)
    for r_ in info["cross_rows"]:
        r_["Waktu_Crossing"] = W.format_waktu(
            W.jam_crossing(mulai, r_["Frame"], info["video_fps"], 1))

    return {
        "Path_Video": rel_path,
        "Set": "PENGEMBANGAN" if gt_day in DEV_DAYS else "UJI",
        "GT_Day": gt_day,
        "GT_User": gt_user,
        "GT_Registered": gt_user in gallery,
        "GT_Action": gt_action,
        "Pred_User": pred_user,
        "Pred_Action": pred_action,
        "Kategori": kategori,
        "Confidence": round(conf, 4),
        "Max_Raw_Score": round(info["max_raw"], 4),
        "Subject_Max_Score": round(info["subject_max"], 4),
        "Subject_Track": info["subject_tid"],
        "Subject_Extent_px": round(info["subject_ext"], 1),
        "Subject_Votes": votes,
        "Subject_Face_Obs": info["subject_face_obs"],
        "Margin_Threshold": margin,
        "Identity_Source": source,
        "Event_Lain": info["event_lain"],
        "Num_Raw_Crossings": info["raw_crossings"],
        "Aturan_Arah": ATURAN_ARAH,
        "Urutan_Crossing": info["urutan_crossing"],
        "Frame_Penentu": info["frame_penentu"],
        "Jam_Mulai_Klip": W.format_waktu(mulai),
        "Jam_Crossing": W.format_waktu(dt),
        "Jam_Aktual_Crossing": W.format_waktu(aktual),
        "Selisih_Waktu_s": round(sel, 2) if sel is not None else None,
        "Cocok_Waktu": (abs(sel) <= TOLERANSI_WAKTU_S) if sel is not None else None,
        "FPS_Fallback": bool(info["fps_fallback"]),
        "Num_Moving_Tracks": info["moving_tracks"],
        "User_Correct": user_ok,
        "Action_Correct": act_ok,
        "Overall_Correct": user_ok and act_ok,
        "Crop_Mode": CROP_MODE,
        "Frames": info["frames_read"],
        "Video_FPS": round(info["video_fps"], 2),
        "Durasi_Video_s": round(duration, 1),
        "Waktu_Proses_s": round(elapsed, 2),
        "FPS_Proses": round(info["frames_read"] / elapsed, 2) if elapsed > 0 else None,
        "RTF": round(elapsed / duration, 2) if duration > 0 else None,
        "T_Decode_s": round(tm["decode"], 2),
        "T_YOLO_s": round(tm["yolo"], 2),
        "T_Face_s": round(tm["face"], 2),
        "T_Anotasi_s": round(tm["annot"], 2),
        "T_Lain_s": round(tm["other"], 2),
        "N_Face_Calls": tm["face_calls"],
        "RAM_Puncak_MB": peak_ram_mb(),
        "Is_Warmup": bool(is_warmup),
        "Valid_Waktu": not SAVE_ANNOTATED_VIDEOS,
        "Track_Summary": info["track_summary"],
        "Video_Output": "",
    }


def finalize_video(raw_path, final_path):
    """Konversi mp4v -> H.264 agar bisa diputar di browser. Waktu TIDAK masuk Waktu_Proses_s."""
    t = time.perf_counter()
    err = ""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [exe, "-y", "-loglevel", "error", "-i", str(raw_path),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
               "-crf", "23", "-movflags", "+faststart", str(final_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(final_path):
            os.remove(raw_path)
            return str(final_path), time.perf_counter() - t, "H.264"
        err = (r.stderr or "")[-300:]
    except Exception as e:
        err = repr(e)
    os.replace(raw_path, final_path)
    return str(final_path), time.perf_counter() - t, f"mp4v (konversi gagal: {err})"