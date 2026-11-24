import streamlit as st
from core import storage, ui_theme

st.set_page_config(
    page_title="Absensi CCTV",
    layout="wide",
    initial_sidebar_state="expanded",
)
storage.ensure_dirs()
ui_theme.inject_css()


def beranda():
    ui_theme.page_header(
        "Absensi CCTV",
        "YOLOv8n, ByteTrack, dan InsightFace. Pemrosesan rekaman, berjalan lokal di CPU, "
        "bukan real-time.",
    )

    db = storage.load_raw_db()
    c1, c2, c3 = st.columns(3)
    c1.metric("Karyawan terdaftar", len(db))
    c2.metric("Total embedding", sum(len(v) for v in db.values()))
    c3.metric(
        "SHA256 galeri",
        storage.sha256_file(storage.GALLERY)[:12] if storage.GALLERY.exists() else "-",
    )

    st.write("")
    ui_theme.section("Alur kerja")
    s1, s2, s3 = st.columns(3)
    with s1.container(border=True):
        ui_theme.card_title(
            "1. Enrollment",
            "Unggah foto wajah tiap karyawan. Galeri disimpan di workspace/gallery.pkl.",
        )
    with s2.container(border=True):
        ui_theme.card_title(
            "2. Proses video",
            "Unggah klip, atur garis dan parameter, isi ground truth, lalu mulai proses.",
        )
    with s3.container(border=True):
        ui_theme.card_title(
            "3. Hasil",
            "Lihat video beranotasi dan tabel kinerja tiap model, lalu unduh Excel.",
        )

    st.write("")
    ui_theme.section("Catatan penelitian")
    st.markdown(
        "- Cadangkan galeri di halaman Enrollment sebelum evaluasi final dan catat SHA256-nya.\n"
        "- Tetapkan parameter pada set pengembangan, lalu kunci untuk set uji.\n"
        "- Waktu proses sudah mencakup overhead UI. Tulis itu sebagai batasan di skripsi."
    )


pg = st.navigation(
    [
        st.Page(beranda, title="Beranda", default=True),
        st.Page("pages/1_Enrollment.py", title="Enrollment"),
        st.Page("pages/2_Proses_Video.py", title="Proses Video"),
    ],
    position="sidebar",
)
pg.run()
