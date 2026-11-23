import numpy as np
import pandas as pd
import streamlit as st

from utils import (CIRI_INTI, STATISTIK, WARNA_GANAS, WARNA_JINAK, daftar_fitur, fmt, kotak,
                   model_prediksi, siapkan_xy)

st.title("Prediksi Satu Pasien")
st.caption("Masukkan 30 ukuran inti sel, lalu model memprediksi jinak (B) atau ganas (M)")

X, y = siapkan_xy()
fitur = daftar_fitur()
modelmodel = model_prediksi()


def isi(nilai):
    for f in fitur:
        st.session_state[f"in_{f}"] = float(nilai[f])


if "in_radius_mean" not in st.session_state:
    isi(X.median())

st.subheader("1. Pilih model dan contoh data")
c1, c2 = st.columns([1.2, 1])
nama_model = c1.selectbox("Model", list(modelmodel.keys()), help="SVC RBF + optimasi adalah model terbaik pada data uji")
ambang = c2.slider("Ambang probabilitas ganas", 0.05, 0.95, 0.50, 0.05,
                   help="Menurunkan ambang (di bawah 0,5) menaikkan recall ganas dengan risiko lebih banyak false positive.")

b1, b2, b3, _ = st.columns([1, 1, 1, 2])
b1.button("Contoh jinak", on_click=lambda: isi(X[y == 0].median()), width="stretch")
b2.button("Contoh ganas", on_click=lambda: isi(X[y == 1].median()), width="stretch")
b3.button("Median semua", on_click=lambda: isi(X.median()), width="stretch")
st.caption("Tombol mengisi semua kolom dengan nilai median kelas yang dipilih. Nilai bisa diubah manual di bawah.")

st.subheader("2. Ukuran inti sel")
tabs = st.tabs(["Mean (rata-rata)", "SE (standard error)", "Worst (terburuk)"])
for tab, stat in zip(tabs, STATISTIK):
    with tab:
        kol = st.columns(5)
        for i, ciri in enumerate(CIRI_INTI):
            f = f"{ciri}_{stat}"
            lo, hi = float(X[f].min()), float(X[f].max())
            kol[i % 5].number_input(f, min_value=0.0, step=max((hi - lo) / 200, 1e-5), format="%.5f", key=f"in_{f}",
                                    help=f"Rentang di dataset: {lo:.5g} sampai {hi:.5g}")

baris = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in fitur}])
luar = [f for f in fitur if not (X[f].min() <= baris.at[0, f] <= X[f].max())]
if luar:
    st.warning(f"{len(luar)} nilai berada di luar rentang data latih ({', '.join(luar[:4])}{'...' if len(luar) > 4 else ''}). "
               "Prediksi bisa kurang andal.", icon=":material/warning:")

st.subheader("3. Hasil prediksi")
if st.button("Prediksi", type="primary"):
    m = modelmodel[nama_model]
    p_ganas = float(m.predict_proba(baris)[0, 1])
    ganas = p_ganas >= ambang
    a, b = st.columns([1, 1.4])
    with a:
        if ganas:
            kotak(f"<h2 style='margin:0;color:{WARNA_GANAS}'>GANAS (M)</h2>Probabilitas ganas {fmt(p_ganas * 100, 1)}%", "sebelum", "HASIL")
        else:
            kotak(f"<h2 style='margin:0;color:{WARNA_JINAK}'>JINAK (B)</h2>Probabilitas ganas {fmt(p_ganas * 100, 1)}%", "sesudah", "HASIL")
    with b:
        st.progress(p_ganas, text=f"Probabilitas ganas {fmt(p_ganas * 100, 1)}% (ambang {fmt(ambang * 100, 0)}%)")
        st.caption(f"Model: {nama_model}")
    mean_cols = [f"{c}_mean" for c in CIRI_INTI]
    banding = pd.DataFrame({"Input": baris.loc[0, mean_cols], "Median jinak": X.loc[y == 0, mean_cols].median(),
                            "Median ganas": X.loc[y == 1, mean_cols].median()})
    st.markdown("**Perbandingan 10 fitur mean dengan median tiap kelas**")
    st.dataframe(banding.style.format("{:.4f}"), width="stretch")
    st.info("Hasil ini murni prediksi model akademik dan bukan diagnosis klinis. Konsultasikan dengan tenaga medis.", icon=":material/info:")
