import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (WARNA_GANAS, WARNA_JINAK, daftar_fitur, fmt, hitung_metrik, model_prediksi,
                   muat_data, tata_letak_plotly)

st.title("Prediksi Massal (CSV)")
st.caption("Unggah berkas CSV berisi banyak pasien, model memprediksi semuanya sekaligus")

fitur = daftar_fitur()
modelmodel = model_prediksi()

with st.expander("Format berkas yang diterima", expanded=True):
    st.markdown(
        "- Wajib memuat **30 kolom fitur** dengan nama seperti pada `wdbc.csv` (mis. `radius_mean`, `concave_points_worst`).\n"
        "- Kolom `id` dan `diagnosis` (B/M atau 0/1) **opsional**. Jika `diagnosis` ada, metrik evaluasi ikut dihitung.\n"
        "- Pemisah koma atau titik koma, nilai desimal memakai titik.")
    contoh = muat_data().sample(10, random_state=1)
    st.download_button("Unduh contoh CSV (10 baris)", contoh.to_csv(index=False).encode("utf-8"),
                       "contoh_wdbc.csv", "text/csv")

c1, c2 = st.columns([1.2, 1])
nama_model = c1.selectbox("Model", list(modelmodel.keys()))
ambang = c2.slider("Ambang probabilitas ganas", 0.05, 0.95, 0.50, 0.05)
berkas = st.file_uploader("Unggah CSV", type=["csv"])


def normalisasi(kolom):
    return [str(c).strip().lower().replace(" ", "_").replace("\ufeff", "") for c in kolom]


if berkas is not None:
    try:
        df = pd.read_csv(berkas, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as e:
        st.error(f"Berkas tidak dapat dibaca: {e}")
        st.stop()
    df.columns = normalisasi(df.columns)
    hilang = [f for f in fitur if f not in df.columns]
    if hilang:
        st.error(f"Kolom fitur tidak lengkap. Kolom yang hilang ({len(hilang)}): {', '.join(hilang[:8])}{'...' if len(hilang) > 8 else ''}")
        st.stop()
    Xin = df[fitur].apply(pd.to_numeric, errors="coerce")
    n_nan = int(Xin.isna().any(axis=1).sum())
    if n_nan:
        st.warning(f"{n_nan} baris mengandung nilai kosong atau bukan angka. Nilai kosong akan diisi median (imputasi di dalam pipeline).")
    m = modelmodel[nama_model]
    p = m.predict_proba(Xin)[:, 1]
    hasil = df.copy()
    hasil["prob_ganas"] = p
    hasil["prediksi"] = np.where(p >= ambang, "M (ganas)", "B (jinak)")

    n_m = int((p >= ambang).sum())
    a, b, c = st.columns(3)
    a.metric("Jumlah pasien", len(hasil))
    b.metric("Diprediksi ganas", n_m)
    c.metric("Diprediksi jinak", len(hasil) - n_m)

    fig = px.histogram(hasil, x="prob_ganas", color="prediksi", nbins=30, barmode="overlay", opacity=0.75,
                       color_discrete_map={"M (ganas)": WARNA_GANAS, "B (jinak)": WARNA_JINAK})
    fig.add_vline(x=ambang, line_dash="dash")
    st.plotly_chart(tata_letak_plotly(fig, 300), width="stretch")
    st.dataframe(hasil, width="stretch", hide_index=True)
    st.download_button("Unduh hasil prediksi (CSV)", hasil.to_csv(index=False).encode("utf-8"), "hasil_prediksi.csv", "text/csv")

    if "diagnosis" in hasil.columns:
        d = hasil["diagnosis"].astype(str).str.strip().str.upper().map({"M": 1, "B": 0, "1": 1, "0": 0})
        if d.isna().any():
            st.warning("Kolom diagnosis berisi nilai selain B/M/0/1, evaluasi dilewati.")
        else:
            st.subheader("Evaluasi terhadap kolom diagnosis")
            met = hitung_metrik(d, (p >= ambang).astype(int), p)
            a, b, c, e = st.columns(4)
            a.metric("Accuracy", fmt(met["Accuracy"]))
            b.metric("Precision (M)", fmt(met["Precision (M)"]))
            c.metric("Recall (M)", fmt(met["Recall (M)"]))
            e.metric("False negative", met["FN"])
            cm = np.array([[met["TN"], met["FP"]], [met["FN"], met["TP"]]])
            fc = px.imshow(cm, text_auto=True, x=["Jinak", "Ganas"], y=["Jinak", "Ganas"], color_continuous_scale="Blues",
                           labels=dict(x="Prediksi", y="Aktual"))
            fc.update_layout(coloraxis_showscale=False)
            st.plotly_chart(tata_letak_plotly(fc, 320), width="stretch")
            st.caption("Jika berkas yang diunggah memuat data yang sama dengan data latih (mis. wdbc.csv utuh), "
                       "metrik ini terlalu optimistis karena sebagian baris sudah dilihat model saat pelatihan.")
else:
    st.info("Unggah berkas CSV untuk memulai, atau unduh contoh format di atas.")
