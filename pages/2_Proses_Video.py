import cv2
import pandas as pd
import streamlit as st
import core.pipeline as P
from core import storage, models, report, ui_results, waktu, ui_theme

storage.ensure_dirs()
ui_theme.page_header(
    "Proses video",
    "Ukur akurasi identitas dan line crossing pada klip rekaman, lalu bandingkan tiap model.",
)

raw_db = storage.load_raw_db()
if not raw_db:
    st.warning("Galeri kosong. Lakukan enrollment dulu di halaman Enrollment.")
    st.stop()
gallery = P.build_gallery(raw_db)
st.caption(f"Galeri: {len(gallery)} user | SHA256 `{storage.sha256_file(storage.GALLERY)[:16]}...`")

# ---------------- 1. Unggah klip ----------------
with st.container(border=True):
    ui_theme.card_title("1. Unggah klip", "Boleh lebih dari satu file. Format: mp4, avi, mov, mkv.")
    files = st.file_uploader(
        "Klip rekaman",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
paths = {}
for f in files or []:
    paths[f.name] = storage.save_upload(f)

# ---------------- 2. Ground truth ----------------
gt_df = None
if files:
    with st.container(border=True):
        ui_theme.card_title(
            "2. Ground truth",
            "Diisi otomatis dari nama file, contoh K017_H1_IN.mp4. Koreksi bila perlu. "
            "Hari H1 dan H2 masuk set PENGEMBANGAN, selain itu set UJI.",
        )
        rows = []
        for f in files:
            u, a, d = P.parse_ground_truth(f.name)
            rows.append({"File": f.name, "GT_User": u, "GT_Action": a, "GT_Day": d,
                         "Jam_Mulai": "", "Jam_Aktual": ""})
        base = pd.DataFrame(rows)
        key = "gt_editor_" + str(abs(hash(tuple(base["File"]))))
        gt_df = st.data_editor(
            base,
            key=key,
            hide_index=True,
            disabled=["File"],
            width="stretch",
            column_config={
                "GT_Action": st.column_config.SelectboxColumn(
                    "GT_Action", options=["MASUK", "KELUAR", "UNKNOWN"]
                ),
                "Jam_Mulai": st.column_config.TextColumn(
                    "Jam_Mulai",
                    help="Jam di frame pertama klip (watermark). Format: 2026-10-29 08:57:38"),
                "Jam_Aktual": st.column_config.TextColumn(
                    "Jam_Aktual",
                    help="Opsional. Jam di watermark saat orang melewati garis (untuk selisih otomatis)."),
            },
        )
        st.caption("Jam_Mulai dan Jam_Aktual opsional, format `YYYY-MM-DD HH:MM:SS`.")

# ---------------- 3. Garis ----------------
with st.container(border=True):
    ui_theme.card_title(
        "3. Garis crossing",
        "Arah MASUK atau KELUAR ditentukan oleh arah garis, dari Start ke End. "
        "Uji dengan satu klip yang arahnya diketahui, lalu tekan Tukar arah bila terbalik.",
    )
    for k, v in (("ls_x", P.LINE_START[0]), ("ls_y", P.LINE_START[1]),
                 ("le_x", P.LINE_END[0]), ("le_y", P.LINE_END[1])):
        st.session_state.setdefault(k, int(v))

    def _swap_line():
        s = st.session_state
        s.ls_x, s.le_x = s.le_x, s.ls_x
        s.ls_y, s.le_y = s.le_y, s.ls_y

    c = st.columns([1, 1, 1, 1, 1.2], vertical_alignment="bottom")
    c[0].number_input("Start X", key="ls_x", step=1)
    c[1].number_input("Start Y", key="ls_y", step=1)
    c[2].number_input("End X", key="le_x", step=1)
    c[3].number_input("End Y", key="le_y", step=1)
    c[4].button("Tukar arah", on_click=_swap_line, width="stretch")

    if paths:
        pick = st.selectbox("Pratinjau pada klip", list(paths))
        cap = cv2.VideoCapture(str(paths[pick]))
        ok, frame0 = cap.read()
        cap.release()
        if ok:
            h0, w0 = frame0.shape[:2]
            s = st.session_state
            p1, p2 = (int(s.ls_x), int(s.ls_y)), (int(s.le_x), int(s.le_y))
            # potongan watermark diambil SEBELUM garis digambar
            wm = frame0[: max(40, int(h0 * 0.10)), : int(w0 * 0.45)].copy()
            k = max(0.6, w0 / P.REF_WIDTH)
            cv2.line(frame0, p1, p2, (0, 255, 255), max(2, int(4 * k)))
            cv2.circle(frame0, p1, max(6, int(14 * k)), (0, 255, 0), -1)
            cv2.circle(frame0, p2, max(6, int(14 * k)), (0, 0, 255), -1)
            cv2.putText(frame0, "START", (p1[0] + 10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2 * k, (0, 255, 0), max(2, int(3 * k)), cv2.LINE_AA)
            cv2.putText(frame0, "END", (p2[0] + 10, p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2 * k, (0, 0, 255), max(2, int(3 * k)), cv2.LINE_AA)
            pv1, pv2 = st.columns([3, 2], gap="medium")
            pv1.image(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB), width="stretch",
                      caption=f"Frame pertama ({w0}x{h0}). Hijau = start, merah = end.")
            pv2.image(cv2.cvtColor(wm, cv2.COLOR_BGR2RGB), width="stretch",
                      caption="Potongan kiri-atas frame pertama, untuk membaca watermark Jam_Mulai.")
        else:
            st.error("Gagal membaca frame pertama.")

# ---------------- 4. Parameter ----------------
with st.container(border=True):
    ui_theme.card_title(
        "4. Parameter",
        "Parameter utama ditampilkan di sini. Parameter lanjutan ada di bagian bawah.",
    )
    a, b, c3 = st.columns(3)
    thr = a.slider("Ambang pengenalan (RECOGNITION_THRESHOLD)", 0.10, 0.90,
                   float(P.RECOGNITION_THRESHOLD), 0.01)
    yconf = b.slider("Confidence YOLO (YOLO_CONF)", 0.05, 0.90, float(P.YOLO_CONF), 0.01)
    imgsz = c3.selectbox("Ukuran input YOLO (YOLO_IMGSZ)", [640, 960, 1280], index=2)

    opsi_arah = ["net", "pertama", "terakhir"]
    d, e, f_ = st.columns(3)
    aturan = d.selectbox(
        "Aturan arah (ATURAN_ARAH)", opsi_arah,
        index=opsi_arah.index(P.ATURAN_ARAH) if P.ATURAN_ARAH in opsi_arah else 0,
        help="net: selisih MASUK-KELUAR, seri = crossing terakhir (perilaku skrip lama). "
             "pertama: crossing pertama. terakhir: crossing terakhir.",
    )
    tol = e.number_input("Toleransi waktu (TOLERANSI_WAKTU_S, detik)", 0.5, 10.0,
                         float(P.TOLERANSI_WAKTU_S), 0.5)
    save_vid = f_.checkbox("Simpan video beranotasi", value=True)
    st.caption("Video beranotasi menambah waktu (anotasi, penulisan, dan konversi H.264). "
               "Untuk pengukuran waktu yang bersih, matikan opsi ini.")

    with st.expander("Parameter lanjutan"):
        g1, g2, g3, g4 = st.columns(4)
        skip = g1.number_input("FACE_REC_SKIP_FRAMES", 1, 30, int(P.FACE_REC_SKIP_FRAMES))
        crop = g2.radio("CROP_MODE", ["body", "head"], horizontal=True,
                        index=0 if P.CROP_MODE == "body" else 1)
        move = g3.number_input("MIN_MOVE_EXTENT (px)", 0, 2000, int(P.MIN_MOVE_EXTENT))
        minface = g4.number_input("MIN_FACE_SIZE (px)", 5, 300, int(P.MIN_FACE_SIZE))

# ---------------- 5. Proses ----------------
start = st.button("Mulai proses", type="primary", disabled=not files)
if start:
    P.RECOGNITION_THRESHOLD = float(thr)
    P.YOLO_CONF = float(yconf)
    P.YOLO_IMGSZ = int(imgsz)
    P.FACE_REC_SKIP_FRAMES = int(skip)
    P.CROP_MODE = crop
    P.MIN_MOVE_EXTENT = int(move)
    P.MIN_FACE_SIZE = int(minface)
    P.SAVE_ANNOTATED_VIDEOS = bool(save_vid)
    P.USE_FALLBACK = False
    P.ATURAN_ARAH = aturan
    P.TOLERANSI_WAKTU_S = float(tol)
    P.LINE_START = (int(st.session_state.ls_x), int(st.session_state.ls_y))
    P.LINE_END = (int(st.session_state.le_x), int(st.session_state.le_y))

    yolo = models.get_yolo()
    face_app = models.get_face_app(P.FACE_DET_SIZE)
    run_dir = storage.new_run_dir()
    runs = []
    st.session_state.setdefault("n_processed", 0)

    def _txt(v):
        s_ = "" if v is None else str(v).strip()
        return "" if s_.lower() in ("nan", "none", "nat") else s_

    total = len(gt_df)
    with st.status("Memproses klip", expanded=True) as status:
        overall = st.progress(0.0, text="Menyiapkan...")
        bar = st.progress(0.0, text="")  # satu bar dipakai ulang untuk setiap klip
        for i, (_, g_row) in enumerate(gt_df.iterrows(), 1):
            name = g_row["File"]
            gt_user = str(g_row["GT_User"]).strip().upper()
            gt_action = str(g_row["GT_Action"]).strip().upper()
            gt_day = str(g_row["GT_Day"]).strip().upper()
            jam_mulai, jam_aktual = _txt(g_row["Jam_Mulai"]), _txt(g_row["Jam_Aktual"])
            for lbl, v in (("Jam_Mulai", jam_mulai), ("Jam_Aktual", jam_aktual)):
                if v and waktu.parse_waktu(v) is None:
                    st.warning(f"{name}: {lbl} '{v}' bukan format YYYY-MM-DD HH:MM:SS, diabaikan.")
            if gt_user in ("", "UNKNOWN") or gt_action not in ("MASUK", "KELUAR"):
                st.error(f"Dilewati: {name}. Ground truth tidak lengkap (user={gt_user}, aksi={gt_action}).")
                continue
            overall.progress((i - 1) / total, text=f"[{i}/{total}] {name}")
            bar.progress(0.0, text=name)

            def cb(fc, tot, fps_now, _bar=bar, _n=name):
                frac = min(1.0, fc / tot) if tot > 0 else 0.0
                _bar.progress(frac, text=f"{_n} | frame {fc}/{tot} | {fps_now:.1f} FPS")

            stem = name.rsplit(".", 1)[0]
            raw_out = run_dir / f"{stem}_raw.mp4" if P.SAVE_ANNOTATED_VIDEOS else None
            try:
                info = P.process_single_video(
                    paths[name], yolo, face_app, gallery,
                    gt_user=gt_user, gt_action=gt_action,
                    raw_out_path=raw_out, progress_cb=cb,
                )
            except Exception as ex:
                st.error(f"Gagal: {name}. {ex!r}")
                continue
            is_warm = st.session_state["n_processed"] == 0
            st.session_state["n_processed"] += 1
            row = P.build_row(info, name, gt_user, gt_action, gt_day, gallery, is_warm,
                              jam_mulai=jam_mulai, jam_aktual=jam_aktual)
            video_path, t_conv, codec = "", 0.0, ""
            if raw_out is not None and raw_out.exists():
                tag = "OK" if row["Overall_Correct"] else "MISMATCH"
                video_path, t_conv, codec = P.finalize_video(raw_out, run_dir / f"{stem}_{tag}.mp4")
            row["Video_Output"] = video_path
            row["T_Konversi_s"] = round(t_conv, 2)
            bar.progress(1.0, text=f"{name} selesai ({row['Waktu_Proses_s']} s) {codec}")
            runs.append({"row": row, "info": info, "video": video_path})
        overall.progress(1.0, text="Selesai")
        status.update(label=f"Selesai: {len(runs)} klip diproses", state="complete", expanded=False)

    st.session_state["runs"] = runs
    st.session_state["cfg_df"] = report.build_config_df(
        P, storage.GALLERY, gallery, len(runs)
    )

# ---------------- 6. Hasil ----------------
runs = st.session_state.get("runs")
if runs:
    st.divider()
    ui_theme.section("Hasil", "Ringkasan seluruh klip pada proses terakhir.")
    ui_results.render_overall(runs, st.session_state.get("cfg_df"))
    st.divider()
    ui_theme.section("Detail per klip")
    names = [r["row"]["Path_Video"] for r in runs]
    sel = st.selectbox("Pilih klip", names)
    ui_results.render_clip(runs[names.index(sel)])
    st.write("")
    if st.button("Bersihkan hasil dari layar"):
        st.session_state.pop("runs", None)
        st.rerun()
