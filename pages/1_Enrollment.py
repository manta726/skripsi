import pickle
import numpy as np
import pandas as pd
import streamlit as st
from core import storage, models, enrollment, ui_theme

storage.ensure_dirs()


@st.dialog("Hapus user dari galeri?", width="small")
def _konfirmasi_hapus(uid, jumlah):
    st.write(f"Seluruh {jumlah} embedding milik {uid} akan dihapus. Tindakan ini tidak dapat dibatalkan.")
    c1, c2 = st.columns(2)
    if c1.button("Batal", width="stretch"):
        st.rerun()
    if c2.button("Hapus", type="primary", width="stretch"):
        db_hapus = storage.load_raw_db()
        db_hapus.pop(uid, None)
        storage.save_raw_db(db_hapus)
        storage.delete_enroll_photos(uid)
        st.session_state["flash_hapus"] = f"{uid} dihapus dari galeri."
        st.rerun()


def _proses_embedding(groups, replace):
    """Proses satu atau banyak karyawan. groups = {ID: [file, ...]}. -> (ringkasan, detail)"""
    face_app = models.get_face_app((640, 640))
    db = storage.load_raw_db()
    ringkasan, detail = [], []
    with st.status("Memproses foto", expanded=True) as status:
        bar = st.progress(0.0, text="Menyiapkan...")
        for i, (uid, fls) in enumerate(groups.items(), 1):
            bar.progress((i - 1) / len(groups), text=f"{uid} ({i}/{len(groups)})")
            results, embs = enrollment.enroll_files(face_app, fls)
            if embs:
                enrollment.add_to_db(db, uid, embs, replace=replace)
                storage.save_enroll_photos(uid, fls, replace=replace)
            for r in results:
                detail.append({"ID": uid, **r})
            ringkasan.append({
                "ID": uid,
                "Foto": len(fls),
                "Berhasil": len(embs),
                "Gagal": len(fls) - len(embs),
                "Total embedding": len(db.get(uid, [])),
            })
        bar.progress(1.0, text="Selesai")
        if any(r["Berhasil"] for r in ringkasan):
            storage.save_raw_db(db)
        status.update(label=f"Selesai: {len(groups)} karyawan diproses", state="complete", expanded=False)
    return ringkasan, detail


ui_theme.page_header(
    "Enrollment wajah",
    "Kelola galeri wajah karyawan yang dipakai sebagai acuan pengenalan.",
)

# Ringkasan ditempatkan di atas, diisi di akhir run agar selalu memakai data terbaru.
ringkasan_slot = st.container()

# Nomor ini dinaikkan setiap proses sukses agar widget upload dan form dikosongkan.
nonce = st.session_state.setdefault("enroll_nonce", 0)

tab_tambah, tab_galeri, tab_cadangan = st.tabs(
    ["Tambah karyawan", "Galeri", "Cadangan dan impor"]
)

# ---------------- Tambah / perbarui ----------------
with tab_tambah:
    # Hasil proses terakhir tetap tampil sampai proses berikutnya.
    last = st.session_state.get("enroll_last")
    if last:
        with st.container(border=True):
            c_msg, c_btn = st.columns([5, 1], vertical_alignment="center")
            c_msg.success(last["judul"])
            if c_btn.button("Tutup hasil", width="stretch"):
                st.session_state.pop("enroll_last", None)
                st.rerun()
            st.dataframe(pd.DataFrame(last["ringkasan"]), hide_index=True)
            with st.expander("Detail per foto"):
                st.dataframe(pd.DataFrame(last["detail"]), hide_index=True)

    mode_input = st.radio(
        "Mode",
        ["Satu karyawan", "Banyak karyawan sekaligus"],
        horizontal=True,
        key=f"mode_tambah_{nonce}",
    )
    mode_ganti = st.radio(
        "Jika ID sudah ada",
        ["Tambahkan ke embedding yang ada", "Ganti seluruh embedding"],
        horizontal=True,
        key=f"rep_{nonce}",
    )
    replace = mode_ganti.startswith("Ganti")

    if mode_input == "Satu karyawan":
        with st.form(f"enroll_form_{nonce}", border=True):
            uid = st.text_input("ID karyawan", placeholder="contoh: K001", key=f"uid_{nonce}")
            imgs = st.file_uploader(
                "Foto wajah (frontal, kiri, kanan, dan seterusnya)",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True,
                key=f"imgs_{nonce}",
            )
            submit = st.form_submit_button("Proses dan simpan", type="primary")

        if submit:
            uid = uid.strip().upper()
            if not uid:
                st.error("ID karyawan wajib diisi.")
            elif not imgs:
                st.error("Unggah minimal satu foto.")
            else:
                if not enrollment.valid_id(uid):
                    st.warning(
                        "ID tidak berformat K + 3 digit. Penguraian ground truth dari nama "
                        "file klip hanya mengenali format itu."
                    )
                ringkasan, detail = _proses_embedding({uid: imgs}, replace)
                if ringkasan[0]["Berhasil"] > 0:
                    st.session_state["enroll_last"] = {
                        "judul": f"{uid}: {ringkasan[0]['Berhasil']} embedding ditambahkan. "
                                 f"Total {ringkasan[0]['Total embedding']}.",
                        "ringkasan": ringkasan,
                        "detail": detail,
                    }
                    st.session_state["enroll_nonce"] = nonce + 1
                    st.rerun()
                else:
                    st.error("Tidak ada embedding yang berhasil dibuat.")
                    st.dataframe(pd.DataFrame(detail), hide_index=True)

    else:
        st.caption(
            "Beri nama file dengan awalan ID karyawan, misalnya K001_depan.jpg, K001_kiri.jpg, "
            "K002_1.png. Semua file dengan awalan yang sama dianggap milik satu karyawan."
        )
        files_batch = st.file_uploader(
            "Foto wajah semua karyawan",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key=f"batch_{nonce}",
        )
        if files_batch:
            groups, invalid = enrollment.group_files_by_id(files_batch)
            prev = pd.DataFrame([
                {"ID": uid, "Jumlah foto": len(fls),
                 "Sudah ada di galeri": "Ya" if uid in storage.load_raw_db() else "Tidak"}
                for uid, fls in sorted(groups.items())
            ])
            c_info = st.columns(2)
            c_info[0].metric("Karyawan terdeteksi", len(groups))
            c_info[1].metric("Foto terbaca", sum(len(v) for v in groups.values()))
            if not prev.empty:
                st.dataframe(prev, hide_index=True)
            if invalid:
                st.warning(
                    f"{len(invalid)} file dilewati karena awalannya bukan ID K + 3 digit: "
                    + ", ".join(invalid)
                )
            if st.button(
                f"Proses {len(groups)} karyawan",
                type="primary",
                disabled=not groups,
                key=f"proses_batch_{nonce}",
            ):
                ringkasan, detail = _proses_embedding(groups, replace)
                total_ok = sum(r["Berhasil"] for r in ringkasan)
                st.session_state["enroll_last"] = {
                    "judul": f"{len(groups)} karyawan diproses. {total_ok} embedding berhasil "
                             f"dari {sum(r['Foto'] for r in ringkasan)} foto.",
                    "ringkasan": ringkasan,
                    "detail": detail,
                }
                st.session_state["enroll_nonce"] = nonce + 1
                st.rerun()

# ---------------- Galeri ----------------
with tab_galeri:
    db = storage.load_raw_db()
    if msg := st.session_state.pop("flash_hapus", None):
        st.success(msg)
    if db:
        tbl = pd.DataFrame(
            [{"User": k, "Jumlah embedding": len(v)} for k, v in sorted(db.items())]
        )
        st.dataframe(tbl, hide_index=True)
        st.caption(
            f"{len(db)} user, {sum(len(v) for v in db.values())} embedding. "
            f"SHA256: `{storage.sha256_file(storage.GALLERY)}`"
        )

        with st.container(border=True):
            ui_theme.card_title(
                "Hapus karyawan",
                "Tindakan ini menghapus seluruh embedding user dari galeri. "
                "Anda akan diminta konfirmasi terlebih dahulu.",
            )
            victim = st.selectbox("Pilih user", sorted(db))
            if st.button("Hapus user"):
                _konfirmasi_hapus(victim, len(db[victim]))

        ui_theme.section(
            "Foto enrollment",
            "Foto asli yang tersimpan di workspace/enrollment. Foto yang didaftarkan sebelum "
            "fitur ini tidak tersedia, karena hanya embedding-nya yang tersimpan.",
        )
        uid_lihat = st.selectbox("Pilih user", sorted(db), key="lihat_foto_uid")
        fotos = storage.list_enroll_photos(uid_lihat)
        if fotos:
            kolom = st.columns(6)
            for i, foto in enumerate(fotos):
                kolom[i % 6].image(str(foto), width="stretch", caption=f"Foto {i + 1}")
        else:
            st.caption("Belum ada foto tersimpan untuk user ini.")
    else:
        st.info("Galeri masih kosong. Tambahkan karyawan pada tab Tambah karyawan.")

# ---------------- Cadangan, ekspor, impor ----------------
with tab_cadangan:
    db = storage.load_raw_db()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            ui_theme.card_title(
                "Cadangan dan ekspor",
                "Simpan salinan galeri di workspace/backups, atau unduh sebagai .pkl.",
            )
            if db:
                if st.button("Buat cadangan galeri (dengan SHA256)"):
                    p = storage.backup_gallery()
                    st.success(f"Tersimpan: {p}")
                    st.code(storage.sha256_file(p))
                st.download_button(
                    "Unduh galeri (.pkl)",
                    data=pickle.dumps(db),
                    file_name="gallery.pkl",
                    mime="application/octet-stream",
                )
            else:
                st.caption("Belum ada galeri untuk dicadangkan.")

    with c2:
        with st.container(border=True):
            ui_theme.card_title(
                "Impor galeri",
                "Impor hanya file .pkl milik Anda sendiri. Pickle dapat menjalankan kode.",
            )
            up = st.file_uploader("File galeri (.pkl)", type=["pkl"], key="import_pkl")
            imp_mode = st.radio(
                "Mode impor",
                ["Gabungkan", "Ganti seluruh galeri"],
                key="imp_mode",
                horizontal=True,
            )
            if up is not None and st.button("Impor"):
                try:
                    new = pickle.loads(up.getvalue())
                    if not isinstance(new, dict):
                        raise ValueError("Isi pkl bukan dict {user: [embedding]}")
                    clean = {}
                    for k, v in new.items():
                        if isinstance(v, np.ndarray) and v.ndim == 1:
                            v = [v]
                        clean[str(k)] = [np.asarray(e, dtype=np.float32).flatten() for e in v]
                    if imp_mode.startswith("Ganti"):
                        db = clean
                    else:
                        for k, v in clean.items():
                            db.setdefault(k, []).extend(v)
                    storage.save_raw_db(db)
                    st.success(f"Impor selesai: {len(clean)} user.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal impor: {e!r}")

# ---------------- Ringkasan (data terbaru) ----------------
db_now = storage.load_raw_db()
with ringkasan_slot:
    c1, c2, c3 = st.columns(3)
    c1.metric("Karyawan terdaftar", len(db_now))
    c2.metric("Total embedding", sum(len(v) for v in db_now.values()))
    c3.metric(
        "SHA256 galeri",
        storage.sha256_file(storage.GALLERY)[:12] if storage.GALLERY.exists() else "-",
    )
