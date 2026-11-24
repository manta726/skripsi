import pickle
import numpy as np
import pandas as pd
import streamlit as st
from core import storage, models, enrollment

st.set_page_config(page_title="Enrollment", layout="wide")
storage.ensure_dirs()
st.title("Enrollment Wajah")

db = storage.load_raw_db()

# ---------------- Tambah / perbarui ----------------
st.subheader("Tambah / perbarui karyawan")
with st.form("enroll_form"):
    uid = st.text_input("ID karyawan (contoh: K001)")
    imgs = st.file_uploader(
        "Foto wajah (frontal, kiri, kanan, dst.)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
    mode = st.radio(
        "Jika ID sudah ada",
        ["Tambahkan ke embedding yang ada", "Ganti seluruh embedding"],
    )
    submit = st.form_submit_button("Proses & simpan")

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

# ---------------- Daftar ----------------
st.subheader("Galeri saat ini")
if db:
    tbl = pd.DataFrame(
        [{"User": k, "Jumlah embedding": len(v)} for k, v in sorted(db.items())]
    )
    st.dataframe(tbl, hide_index=True)
    st.caption(
        f"{len(db)} user, {sum(len(v) for v in db.values())} embedding. "
        f"SHA256: `{storage.sha256_file(storage.GALLERY)}`"
    )
else:
    st.info("Galeri masih kosong.")

# ---------------- Hapus ----------------
if db:
    with st.expander("Hapus karyawan"):
        victim = st.selectbox("Pilih user", sorted(db))
        sure = st.checkbox("Saya yakin ingin menghapus user ini")
        if st.button("Hapus", disabled=not sure):
            db.pop(victim, None)
            storage.save_raw_db(db)
            st.success(f"{victim} dihapus.")
            st.rerun()

# ---------------- Cadangan, ekspor, impor ----------------
st.subheader("Cadangan, ekspor, dan impor")
c1, c2 = st.columns(2)
with c1:
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
with c2:
    up = st.file_uploader("Impor galeri (.pkl)", type=["pkl"], key="import_pkl")
    imp_mode = st.radio("Mode impor", ["Gabungkan", "Ganti seluruh galeri"], key="imp_mode")
    st.caption("Impor hanya file .pkl milik Anda sendiri. Pickle dapat menjalankan kode.")
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