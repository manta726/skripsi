import py_compile
import re
import shutil
import sys
from pathlib import Path

p = Path("core/pipeline.py")
src = p.read_text(encoding="utf-8")
if "ATURAN_ARAH" in src:
    sys.exit("Sudah pernah ditambal (ATURAN_ARAH ditemukan). Tidak ada perubahan.")

# ---- penggantian teks persis ----
EDITS = []

EDITS.append((
'''    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0
''',
'''    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_fallback = False
    if fps <= 0:
        fps = 20.0
        fps_fallback = True
'''))

EDITS.append((
'''    events = []
    subject_tid, subject_ext = None, 0.0
    if candidates:
        subject_tid = max(candidates, key=lambda t_: track_extent(track_stats[t_]))
        subject_ext = track_extent(track_stats[subject_tid])
        evs = by_track[subject_tid]
        net = sum(1 if e["status"] == "MASUK" else -1 for e in evs)
        if net > 0:
            status = "MASUK"
        elif net < 0:
            status = "KELUAR"
        else:
            status = evs[-1]["status"]
''',
'''    events = []
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
'''))

EDITS.append((
'''        "video_fps": fps,
''',
'''        "video_fps": fps,
        "fps_fallback": fps_fallback,
        "urutan_crossing": urutan,
        "frame_penentu": frame_penentu,
'''))

EDITS.append((
'''def build_row(info, rel_path, gt_user, gt_action, gt_day, gallery, is_warmup):
''',
'''def build_row(info, rel_path, gt_user, gt_action, gt_day, gallery, is_warmup,
              jam_mulai="", jam_aktual=""):
'''))

EDITS.append((
'''    margin = round(conf - RECOGNITION_THRESHOLD, 4) if votes > 0 else None
''',
'''    margin = round(conf - RECOGNITION_THRESHOLD, 4) if votes > 0 else None
    mulai = W.parse_waktu(jam_mulai)
    aktual = W.parse_waktu(jam_aktual)
    fp = info.get("frame_penentu")
    dt = W.jam_crossing(mulai, fp, info["video_fps"], 1) if fp is not None else None
    sel = W.selisih_detik(dt, aktual)
    for r_ in info["cross_rows"]:
        r_["Waktu_Crossing"] = W.format_waktu(
            W.jam_crossing(mulai, r_["Frame"], info["video_fps"], 1))
'''))

EDITS.append((
'''        "Num_Raw_Crossings": info["raw_crossings"],
''',
'''        "Num_Raw_Crossings": info["raw_crossings"],
        "Aturan_Arah": ATURAN_ARAH,
        "Urutan_Crossing": info["urutan_crossing"],
        "Frame_Penentu": info["frame_penentu"],
        "Jam_Mulai_Klip": W.format_waktu(mulai),
        "Jam_Crossing": W.format_waktu(dt),
        "Jam_Aktual_Crossing": W.format_waktu(aktual),
        "Selisih_Waktu_s": round(sel, 2) if sel is not None else None,
        "Cocok_Waktu": (abs(sel) <= TOLERANSI_WAKTU_S) if sel is not None else None,
        "FPS_Fallback": bool(info["fps_fallback"]),
'''))

for old, new in EDITS:
    n = src.count(old)
    if n != 1:
        sys.exit(f"GAGAL: teks berikut ditemukan {n}x (harus 1x). Tidak ada file diubah.\n---\n{old}")
    src = src.replace(old, new)

# ---- import dan parameter (disisipkan setelah baris yang ada) ----
m = re.search(r"^import cv2[^\n]*\n", src, flags=re.M)
if not m:
    sys.exit("GAGAL: baris 'import cv2' tidak ditemukan. Tidak ada file diubah.")
src = src[:m.end()] + "from core import waktu as W\n" + src[m.end():]

m = re.search(r"^MIN_MOVE_EXTENT\s*=[^\n]*\n", src, flags=re.M)
if not m:
    sys.exit("GAGAL: baris 'MIN_MOVE_EXTENT = ...' tidak ditemukan. Tidak ada file diubah.")
src = (src[:m.end()]
       + 'ATURAN_ARAH = "net"        # "net" | "pertama" | "terakhir"\n'
       + "TOLERANSI_WAKTU_S = 2.0\n"
       + src[m.end():])

shutil.copy(p, "core/pipeline.py.bak")
p.write_text(src, encoding="utf-8")
py_compile.compile(str(p), doraise=True)
print("SELESAI: pipeline.py ditambal, cadangan di core/pipeline.py.bak, py_compile OK")