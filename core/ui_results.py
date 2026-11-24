import numpy as np
import pandas as pd
import streamlit as st
from core import report


def render_overall(runs, cfg_df):
    df = pd.DataFrame([r["row"] for r in runs])
    summary, m = report.build_summary(df)
    c = st.columns(5)
    c[0].metric("Klip", m["n"])
    c[1].metric("Akurasi identitas", f"{m['acc_user']:.1f}%")
    c[2].metric("Akurasi line crossing", f"{m['acc_action']:.1f}%")
    c[3].metric("End-to-end", f"{m['acc_overall']:.1f}%")
    c[4].metric("CI95 Wilson", f"{m['lo']:.1f}-{m['hi']:.1f}%")
    if (df["Valid_Waktu"] == False).any():  # noqa: E712
        st.caption("Sebagian klip memakai video beranotasi, sehingga waktunya tidak valid "
                   "untuk statistik waktu. Klip pertama adalah warm-up.")
    st.download_button(
        "Unduh Excel",
        data=report.build_excel(runs, cfg_df),
        file_name="hasil_evaluasi_streamlit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    tab_model, tab_rekap, tab_kat, tab_lain = st.tabs(
        ["Kinerja model", "Rekap absensi", "Kategori", "Ringkasan dan konfigurasi"]
    )
    with tab_model:
        st.caption("Kinerja tiap model dihitung per klip.")
        st.dataframe(report.model_perf_df(runs), hide_index=True)
    with tab_rekap:
        rekap = report.rekap_absensi(df)
        if rekap.empty:
            st.info("Rekap absensi belum tersedia.")
        else:
            st.caption("Hasil sistem, per user per tanggal.")
            st.dataframe(rekap, hide_index=True)
    with tab_kat:
        st.caption("Jumlah klip per set dan kategori hasil.")
        st.dataframe(pd.crosstab(df["Set"], df["Kategori"], margins=True))
    with tab_lain:
        st.caption("Ringkasan lengkap")
        st.dataframe(summary, hide_index=True)
        if cfg_df is not None:
            st.caption("Konfigurasi run (masuk ke sheet Konfigurasi)")
            st.dataframe(cfg_df, hide_index=True)


def render_clip(run):
    row, info = run["row"], run["info"]
    tabs = st.tabs(["Video & Hasil", "YOLOv8n", "ByteTrack", "InsightFace", "Waktu"])

    # ---- Video & Hasil ----
    with tabs[0]:
        ok = row["Overall_Correct"]
        (st.success if ok else st.error)(
            f"{'BENAR' if ok else 'SALAH'}: kategori `{row['Kategori']}`"
        )
        c = st.columns(4)
        c[0].metric("GT", f"{row['GT_User']} / {row['GT_Action']}")
        c[1].metric("Prediksi", f"{row['Pred_User']} / {row['Pred_Action']}")
        c[2].metric("Confidence", f"{row['Confidence']:.3f}")
        c[3].metric("Max skor mentah", f"{row['Max_Raw_Score']:.3f}")
        st.markdown(f"**Urutan crossing:** `{row['Urutan_Crossing'] or '-'}` "
                    f"(aturan: {row['Aturan_Arah']})")
        if row["Jam_Crossing"]:
            msg = f"**Jam crossing penentu:** `{row['Jam_Crossing']}` (frame {row['Frame_Penentu']})"
            if row["Selisih_Waktu_s"] is not None:
                msg += (f" | selisih vs watermark: {row['Selisih_Waktu_s']} s "
                        f"({'cocok' if row['Cocok_Waktu'] else 'TIDAK cocok'})")
            st.markdown(msg)
        if row["FPS_Fallback"]:
            st.warning("Metadata FPS tidak terbaca, dipakai 20,0. Jam crossing tidak dapat dipercaya.")
        if run.get("video"):
            st.video(run["video"])
        else:
            st.info("Video beranotasi tidak disimpan untuk klip ini.")
        st.caption(f"Ringkasan track: {row['Track_Summary']}")

    # ---- YOLOv8n ----
    with tabs[1]:
        dl = pd.DataFrame(info["det_log"])
        n = max(info["frames_read"], 1)
        if dl.empty:
            st.info("Tidak ada deteksi person yang berhasil dilacak.")
        else:
            c = st.columns(4)
            c[0].metric("Deteksi (ber-ID)", len(dl))
            c[1].metric("Rata-rata per frame", f"{len(dl)/n:.2f}")
            c[2].metric("Conf rata-rata", f"{dl['Conf'].mean():.3f}")
            c[3].metric("ms/frame (YOLO+ByteTrack)",
                        f"{1000*info['timing']['yolo']/n:.0f}")
            st.caption("Jumlah deteksi per frame")
            st.line_chart(pd.DataFrame({"Deteksi per frame": info["dets_per_frame"]}))
            st.caption("Distribusi confidence deteksi")
            hist, edges = np.histogram(dl["Conf"], bins=10, range=(0.0, 1.0))
            st.bar_chart(pd.DataFrame(
                {"Jumlah": hist},
                index=[f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(hist))]))
            with st.expander("Log deteksi (2000 baris pertama)"):
                st.dataframe(dl.head(2000), hide_index=True)
        st.caption("Waktu YOLO dan ByteTrack berasal dari satu panggilan track() sehingga "
                   "tidak dapat dipisah. Precision/recall deteksi memerlukan anotasi kotak "
                   "dan tidak dihitung di sini.")

    # ---- ByteTrack ----
    with tabs[2]:
        tr = pd.DataFrame(info["track_rows"])
        c = st.columns(4)
        c[0].metric("Jumlah track", len(tr))
        c[1].metric("Track bergerak", info["moving_tracks"])
        c[2].metric("Crossing mentah", info["raw_crossings"])
        c[3].metric("Subjek", f"T{info['subject_tid']}" if info["subject_tid"] is not None else "-")
        if not tr.empty:
            st.caption("Jumlah frame terlihat dan jangkauan gerak per track")
            st.bar_chart(tr.set_index("Track")[["Frames_Seen", "Extent_px"]])
            st.dataframe(tr, hide_index=True)
        cr = pd.DataFrame(info["cross_rows"])
        st.caption("Kejadian crossing")
        if cr.empty:
            st.info("Tidak ada crossing.")
        else:
            st.dataframe(cr, hide_index=True)

    # ---- InsightFace ----
    with tabs[3]:
        ob = pd.DataFrame(info["obs_log"])
        calls = info["timing"]["face_calls"]
        c = st.columns(4)
        c[0].metric("Panggilan wajah", calls)
        c[1].metric("Observasi wajah", len(ob))
        c[2].metric("Hit rate", f"{100*len(ob)/calls:.0f}%" if calls else "-")
        c[3].metric("ms/panggilan", f"{1000*info['timing']['face']/calls:.0f}" if calls else "-")
        if ob.empty:
            st.info("Tidak ada wajah yang terdeteksi/dicocokkan.")
        else:
            agg = ob.groupby("Track").agg(
                Observasi=("Frame", "size"),
                Skor_rata2=("Best_Score", "mean"),
                Skor_maks=("Best_Score", "max"),
                DetScore_rata2=("Det_Score", "mean"),
                User_terbanyak=("Best_User", lambda s: s.value_counts().index[0]),
            ).round(4).reset_index()
            st.dataframe(agg, hide_index=True)
            piv = ob.pivot_table(index="Frame", columns="Track",
                                 values="Best_Score", aggfunc="mean")
            piv.columns = [f"T{c_}" for c_ in piv.columns]
            st.caption("Skor cosine terbaik per frame, satu garis per track")
            st.line_chart(piv)
            st.caption("Skor cosine terbaik per observasi. Observasi dengan skor di bawah "
                       "threshold dicatat sebagai Unknown pada pemungutan suara track.")
            with st.expander("Log observasi (2000 baris pertama)"):
                st.dataframe(ob.head(2000), hide_index=True)

    # ---- Waktu ----
    with tabs[4]:
        tm = info["timing"]
        comp = pd.DataFrame({
            "Komponen": ["Decode", "YOLOv8n+ByteTrack", "InsightFace", "Anotasi video", "Lain-lain"],
            "Detik": [tm["decode"], tm["yolo"], tm["face"], tm["annot"], tm["other"]],
        })
        total = max(comp["Detik"].sum(), 1e-9)
        comp["%"] = (100 * comp["Detik"] / total).round(1)
        comp["Detik"] = comp["Detik"].round(2)
        c = st.columns(4)
        c[0].metric("Waktu proses", f"{row['Waktu_Proses_s']} s")
        c[1].metric("FPS proses", row["FPS_Proses"])
        c[2].metric("RTF", row["RTF"])
        c[3].metric("RAM puncak", f"{row['RAM_Puncak_MB']} MB")
        st.dataframe(comp, hide_index=True)
        st.caption("Waktu per komponen (detik)")
        st.bar_chart(comp.set_index("Komponen")["Detik"])
        flags = []
        if row["Is_Warmup"]:
            flags.append("klip ini adalah warm-up (jangan masuk statistik waktu)")
        if not row["Valid_Waktu"]:
            flags.append("video beranotasi aktif, waktu tidak valid untuk data RQ3")
        if flags:
            st.warning("; ".join(flags))
        st.caption("Waktu konversi H.264 dicatat terpisah (kolom T_Konversi_s) dan tidak "
                   "termasuk Waktu_Proses_s. Overhead progress bar UI termasuk di dalamnya.")