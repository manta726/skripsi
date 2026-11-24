import hashlib
import importlib.metadata
import io
import os
import platform
from datetime import datetime
import numpy as np
import pandas as pd
from core.pipeline import wilson

CAT_ORDER = ["benar", "wrong_direction", "unknown", "no_crossing", "wrong_identity"]


def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "?"


def _ver(pkg):
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return "?"


def _bytetrack_params():
    try:
        import ultralytics
        import yaml
        p = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "bytetrack.yaml")
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def build_config_df(P, gallery_path, gallery, n_videos):
    g = lambda name: getattr(P, name, "?")
    rows = [
        ("Waktu run", datetime.now().isoformat(timespec="seconds")),
        ("Aplikasi", "Streamlit (cctv_streamlit)"),
        ("Jumlah video di run ini", n_videos),
        ("Galeri", str(gallery_path)),
        ("Galeri SHA256", _sha256(gallery_path)),
        ("Jumlah user di galeri", len(gallery)),
        ("User terdaftar", ", ".join(sorted(gallery.keys()))),
        ("RECOGNITION_THRESHOLD", g("RECOGNITION_THRESHOLD")),
        ("YOLO_CONF", g("YOLO_CONF")),
        ("YOLO_IMGSZ", g("YOLO_IMGSZ")),
        ("YOLO_DEVICE", g("YOLO_DEVICE")),
        ("Model deteksi", "yolov8n.pt, kelas 0 (person)"),
        ("Model wajah", "buffalo_l, CPUExecutionProvider, modul: detection+recognition"),
        ("Tracker", "bytetrack.yaml (bawaan Ultralytics)"),
        ("FACE_REC_SKIP_FRAMES", g("FACE_REC_SKIP_FRAMES")),
        ("FACE_DET_SIZE", g("FACE_DET_SIZE")),
        ("CROP_MODE", g("CROP_MODE")),
        ("HEAD_FRAC / HEAD_PAD / HEAD_TOP_PAD",
         f"{g('HEAD_FRAC')} / {g('HEAD_PAD')} / {g('HEAD_TOP_PAD')}"),
        ("MIN_FACE_SIZE", g("MIN_FACE_SIZE")),
        ("MIN_MOVE_EXTENT", g("MIN_MOVE_EXTENT")),
        ("USE_FALLBACK", g("USE_FALLBACK")),
        ("ATURAN_ARAH", g("ATURAN_ARAH")),
        ("TOLERANSI_WAKTU_S", g("TOLERANSI_WAKTU_S")),
        ("Rumus jam crossing",
         "Jam mulai klip (input manual dari watermark) + (Frame-1)/Video_FPS; Frame mulai dari 1"),
        ("Simpan video berlabel", g("SAVE_ANNOTATED_VIDEOS")),
        ("LINE_START", str(g("LINE_START"))),
        ("LINE_END", str(g("LINE_END"))),
        ("Anchor garis", "BOTTOM_CENTER"),
        ("Titik jelajah gerak", "top-center bounding box"),
        ("Python", platform.python_version()),
        ("OS", platform.platform()),
        ("CPU (platform.processor)", platform.processor()),
        ("Thread logis (os.cpu_count)", os.cpu_count()),
    ]
    try:
        import torch
        rows.append(("torch.cuda.is_available", torch.cuda.is_available()))
    except Exception:
        pass
    for pkg in ("torch", "ultralytics", "supervision", "insightface", "onnxruntime",
                "opencv-python", "numpy", "pandas", "openpyxl", "streamlit", "psutil",
                "imageio-ffmpeg"):
        rows.append((f"versi {pkg}", _ver(pkg)))
    for k, v in _bytetrack_params().items():
        rows.append((f"bytetrack.yaml: {k}", v))
    return pd.DataFrame([(a, str(b)) for a, b in rows], columns=["Parameter", "Nilai"])


def group_table(df, keys):
    tmp = df.assign(_ok=(df["Kategori"] == "benar"))
    t = tmp.groupby(keys).agg(Klip=("_ok", "size"), Benar=("_ok", "sum")).reset_index()
    t["Benar"] = t["Benar"].astype(int)
    t["Akurasi_%"] = (t["Benar"] / t["Klip"] * 100).round(1)
    cis = [wilson(int(k), int(n)) for k, n in zip(t["Benar"], t["Klip"])]
    t["CI95_bawah_%"] = [round(c[0], 1) for c in cis]
    t["CI95_atas_%"] = [round(c[1], 1) for c in cis]
    return t


def build_summary(df):
    n = len(df)
    acc_user = df["User_Correct"].sum() / n * 100
    acc_action = df["Action_Correct"].sum() / n * 100
    acc_overall = df["Overall_Correct"].sum() / n * 100
    lo, hi = wilson(int(df["Overall_Correct"].sum()), n)
    claimed = df[~df["Pred_User"].isin(["Unknown", "NO_DETECTION"])]
    n_fa = int((~claimed["User_Correct"]).sum())
    n_unknown = int((df["Pred_User"] == "Unknown").sum())
    n_nocross = int((df["Pred_Action"] == "NO_CROSSING").sum())
    prec = (claimed["User_Correct"].sum() / len(claimed) * 100) if len(claimed) else 0.0
    n_unreg = int((~df["GT_Registered"]).sum())
    tdf = df[df["Valid_Waktu"] & ~df["Is_Warmup"]]
    nan = float("nan")
    mean_t = float(tdf["Waktu_Proses_s"].mean()) if len(tdf) else nan
    med_t = float(tdf["Waktu_Proses_s"].median()) if len(tdf) else nan
    sd_t = float(tdf["Waktu_Proses_s"].std()) if len(tdf) > 1 else nan
    mean_rtf = float(tdf["RTF"].mean()) if len(tdf) else nan
    sel_s = pd.to_numeric(df["Selisih_Waktu_s"], errors="coerce").dropna()
    n_w = len(sel_s)
    n_w_ok = int((df["Cocok_Waktu"] == True).sum())  # noqa: E712
    rows = [
        ("Total klip", n),
        ("GT tidak terdaftar", n_unreg),
        ("Akurasi identitas (%)", round(acc_user, 2)),
        ("Precision identitas (%)", round(prec, 2)),
        ("Salah identitas / FA (jumlah)", n_fa),
        ("Unknown / FR (jumlah)", n_unknown),
        ("Tanpa crossing (jumlah)", n_nocross),
        ("Akurasi line crossing (%)", round(acc_action, 2)),
        ("Akurasi end-to-end (%)", round(acc_overall, 2)),
        ("CI95 Wilson end-to-end bawah (%)", round(lo, 1)),
        ("CI95 Wilson end-to-end atas (%)", round(hi, 1)),
        ("Waktu pemrosesan jumlah (detik)", round(float(df["Waktu_Proses_s"].sum()), 1)),
        ("Waktu rata-rata per klip, valid (detik)", round(mean_t, 2)),
        ("Waktu median per klip, valid (detik)", round(med_t, 2)),
        ("Waktu SD per klip, valid (detik)", round(sd_t, 2)),
        ("RTF rata-rata, valid", round(mean_rtf, 2)),
        ("Jumlah klip untuk statistik waktu", len(tdf)),
        ("Klip dengan jam aktual terisi", n_w),
        ("Cocok waktu (%)", round(n_w_ok / n_w * 100, 1) if n_w else nan),
        ("Median |selisih waktu| (detik)", round(float(sel_s.abs().median()), 2) if n_w else nan),
        ("Maks |selisih waktu| (detik)", round(float(sel_s.abs().max()), 2) if n_w else nan),
    ]
    m = {"n": n, "acc_user": acc_user, "acc_action": acc_action,
         "acc_overall": acc_overall, "lo": lo, "hi": hi}
    return pd.DataFrame(rows, columns=["Metrik", "Nilai"]), m


def rekap_absensi(df):
    """Rekap hasil sistem: jam masuk = MASUK paling awal, jam pulang = KELUAR paling akhir."""
    d = df[(df["Jam_Crossing"].astype(str) != "")
           & ~df["Pred_User"].isin(["Unknown", "NO_DETECTION"])].copy()
    if d.empty:
        return pd.DataFrame()
    d["Tanggal"] = d["Jam_Crossing"].str[:10]
    keys = ["Pred_User", "Tanggal"]
    masuk = d[d["Pred_Action"] == "MASUK"].groupby(keys)["Jam_Crossing"].min()
    pulang = d[d["Pred_Action"] == "KELUAR"].groupby(keys)["Jam_Crossing"].max()
    return pd.concat({"Jam_Masuk": masuk, "Jam_Pulang": pulang}, axis=1).reset_index()


def model_perf_df(runs):
    rows = []
    for r in runs:
        info, row, tm = r["info"], r["row"], r["info"]["timing"]
        n = max(info["frames_read"], 1)
        dl, obs = info["det_log"], info["obs_log"]
        calls = tm["face_calls"]
        rows.append({
            "Path_Video": row["Path_Video"],
            "Frames": info["frames_read"],
            "YOLO+Track_s": round(tm["yolo"], 2),
            "YOLO_ms/frame": round(1000 * tm["yolo"] / n, 1),
            "Deteksi_berID": len(dl),
            "Deteksi/frame": round(len(dl) / n, 2),
            "Conf_rata2": round(float(np.mean([d["Conf"] for d in dl])), 3) if dl else None,
            "Jumlah_Track": len(info["track_rows"]),
            "Track_Gerak": info["moving_tracks"],
            "Crossing_Mentah": info["raw_crossings"],
            "Face_s": round(tm["face"], 2),
            "Face_Calls": calls,
            "Face_ms/call": round(1000 * tm["face"] / calls, 1) if calls else None,
            "Face_Obs": len(obs),
            "Face_Hit_%": round(100 * len(obs) / calls, 1) if calls else None,
            "DetScore_rata2": round(float(np.mean([o["Det_Score"] for o in obs])), 3) if obs else None,
            "Skor_Cocok_rata2": round(float(np.mean([o["Best_Score"] for o in obs])), 3) if obs else None,
            "Decode_s": round(tm["decode"], 2),
            "Anotasi_s": round(tm["annot"], 2),
            "Lain_s": round(tm["other"], 2),
            "Waktu_Proses_s": row["Waktu_Proses_s"],
            "FPS_Proses": row["FPS_Proses"],
            "RTF": row["RTF"],
            "Is_Warmup": row["Is_Warmup"],
            "Valid_Waktu": row["Valid_Waktu"],
        })
    return pd.DataFrame(rows)


def _concat(runs, key):
    frames = []
    for r in runs:
        d = pd.DataFrame(r["info"][key])
        if not d.empty:
            d.insert(0, "Path_Video", r["row"]["Path_Video"])
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_excel(runs, cfg_df):
    df = pd.DataFrame([r["row"] for r in runs])
    summary, _ = build_summary(df)
    sheets = {
        "Hasil": (df, False),
        "Ringkasan": (summary, False),
        "Kategori": (pd.crosstab(df["Set"], df["Kategori"], margins=True), True),
        "Per_Set": (group_table(df, ["Set"]), False),
        "Per_Hari": (group_table(df, ["Set", "GT_Day"]), False),
        "Per_Karyawan": (group_table(df, ["GT_User"]), False),
        "Kinerja_Model": (model_perf_df(runs), False),
        "CM_Aksi": (pd.crosstab(df["GT_Action"], df["Pred_Action"]), True),
        "CM_Identitas": (pd.crosstab(df["GT_User"], df["Pred_User"]), True),
        "Track": (_concat(runs, "track_rows"), False),
        "Crossing": (_concat(runs, "cross_rows"), False),
        "Rekap_Absensi": (rekap_absensi(df), False),
        "Observasi_Wajah": (_concat(runs, "obs_log"), False),
        "Deteksi_YOLO": (_concat(runs, "det_log"), False),
    }
    if cfg_df is not None:
        sheets["Konfigurasi"] = (cfg_df, False)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, (frame, keep_index) in sheets.items():
            if frame is None or len(frame) == 0:
                continue
            frame.to_excel(w, sheet_name=name, index=keep_index)
    return buf.getvalue()