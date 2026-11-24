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
        st.session_state["flash_hapus"] = f"{uid} dihapus dari galeri."
        st.rerun()


ui_theme.page_header(
    "Enrollment wajah",
    "Kelola galeri wajah karyawan yang dipakai sebagai acuan pengenalan.",
)

# Ringkasan ditempatkan di atas, diisi di akhir run agar selalu memakai data terbaru.
ringkasan = st.container()

tab_tambah, tab_galeri, tab_cadangan = st.tabs(
    ["Tambah karyawan", "Galeri", "Cadangan dan impor"]
)

# ---------------- Tambah / perbarui ----------------
with tab_tambah:
    db = storage.load_raw_db()
    with st.form("enroll_form", border=True):
        uid = st.text_input("ID karyawan", placeholder="contoh: K001")
        imgs = st.file_uploader(
            "Foto wajah (frontal, kiri, kanan, dan seterusnya)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
        mode = st.radio(
            "Jika ID sudah ada",
            ["Tambahkan ke embedding yang ada", "Ganti seluruh embedding"],
            horizontal=True,
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
            face_app = models.get_face_app((640, 640))
            results, embs = enrollment.enroll_files(face_app, imgs)
            st.dataframe(pd.DataFrame(results), hide_index=True)
            if embs:
                enrollment.add_to_db(db, uid, embs, replace=mode.startswith("Ganti"))
                storage.save_raw_db(db)
                st.success(f"{uid}: {len(embs)} embedding ditambahkan. Total {len(db[uid])}.")
            else:
                st.error("Tidak ada embedding yang berhasil dibuat.")

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
with ringkasan:
    c1, c2, c3 = st.columns(3)
    c1.metric("Karyawan terdaftar", len(db_now))
    c2.metric("Total embedding", sum(len(v) for v in db_now.values()))
    c3.metric(
        "SHA256 galeri",
        storage.sha256_file(storage.GALLERY)[:12] if storage.GALLERY.exists() else "-",
    )
