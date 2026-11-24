import streamlit as st
from core import storage

st.set_page_config(page_title="Absensi CCTV", layout="wide")
storage.ensure_dirs()

st.title("Absensi CCTV: YOLOv8n + ByteTrack + InsightFace")
st.caption("Pemrosesan rekaman (bukan real-time), berjalan lokal di CPU.")

db = storage.load_raw_db()
c1, c2, c3 = st.columns(3)
c1.metric("Karyawan terdaftar", len(db))
c2.metric("Total embedding", sum(len(v) for v in db.values()))
c3.metric("SHA256 galeri", storage.sha256_file(storage.GALLERY)[:12] if storage.GALLERY.exists() else "-")

st.markdown(
    """
### Alur kerja
1. **Enrollment**: unggah foto wajah tiap karyawan. Galeri disimpan di `workspace/gallery.pkl`.
2. **Proses Video**: unggah klip, atur garis dan parameter, isi ground truth, lalu mulai proses.
3. Lihat video beranotasi dan tabel kinerja tiap model, lalu unduh Excel.

### Catatan
- Cadangkan galeri (tombol di halaman Enrollment) sebelum evaluasi final dan catat SHA256-nya.
- Tetapkan parameter pada set pengembangan, lalu kunci untuk set uji.
- Waktu proses di aplikasi ini sudah mencakup overhead UI. Tulis itu sebagai batasan di skripsi.
"""
)