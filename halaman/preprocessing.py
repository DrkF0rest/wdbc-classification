import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler

from utils import (WARNA_GANAS, WARNA_JINAK, bagi_data, fmt, kotak, muat_data,
                   ringkas_kualitas_data, siapkan_xy, tata_letak_plotly)

st.title("Preprocessing")
st.caption("Data dibersihkan agar model tidak menghitung dobel dan tidak menghafal")

q = ringkas_kualitas_data()
X, y = siapkan_xy()
Xtr, Xte, ytr, yte = bagi_data()

t1, t2, t3, t4 = st.tabs(["Langkah cleaning", "Sebelum vs sesudah", "Scaling", "Split & pipeline"])

with t1:
    tabel = pd.DataFrame([
        ("Hapus kolom id", "569 id unik (rasio 1,00)", "Dibuang", "Cegah model menghafal ID"),
        ("Cek duplikat", f"{q['duplikat']} baris duplikat", "Tidak ada yang dihapus", "Cegah sampel terhitung dobel"),
        ("Cek data kosong", f"{q['missing']} missing", "Imputasi median di dalam pipeline", "Aman jika ada data kosong"),
        ("Cek anomali fisik", f"{q['anomali']} baris nilai <= 0", "Tidak ada yang dibuang", "Ukuran sel harus positif"),
        ("Kode target", "B / M (teks)", "B = 0, M = 1 (ganas positif)", "Model butuh target numerik"),
        ("Outlier", f"{q['outlier_baris']} baris (IQR)", "Dipertahankan", "Valid biologis, banyak kasus ganas"),
        ("Skala fitur", "Rentang sangat berbeda", "StandardScaler dalam pipeline", "K-NN dan SVC sensitif skala"),
    ], columns=["Langkah", "Kondisi data", "Tindakan", "Alasan"])
    st.dataframe(tabel, width="stretch", hide_index=True)
    st.caption("Cleaning memastikan tiap pasien terhitung sekali, ID tidak dihafal model, dan preprocessing tidak bocor. "
               "Pada dataset ini tidak ada baris yang perlu dihapus.")

with t2:
    df = muat_data()
    c1, c2 = st.columns(2)
    with c1:
        kotak(f"<h3 style='margin:0'>{df.shape[0]} x {df.shape[1]}</h3>id, diagnosis (teks B/M), dan 30 fitur", "sebelum", "SEBELUM")
        st.dataframe(df.head(5), width="stretch", hide_index=True)
    with c2:
        kotak(f"<h3 style='margin:0'>{X.shape[0]} x {X.shape[1]}</h3>30 fitur numerik + target biner (0 = jinak, 1 = ganas)", "sesudah", "SESUDAH")
        st.dataframe(X.assign(target=y).head(5), width="stretch", hide_index=True)
    a, b, c = st.columns(3)
    a.metric("Baris dibuang", 0)
    b.metric("Data latih : data uji", f"{len(Xtr)} : {len(Xte)}", help="Split 80/20 stratified")
    c.metric("Scaler dan imputer", "Dalam pipeline", help="Di-fit hanya pada data latih")

with t3:
    st.write("Fitur WDBC memiliki disparitas skala sekitar **140.000x** antar fitur (mis. area vs fractal dimension). "
             "K-NN dan SVC sensitif terhadap skala sehingga **StandardScaler** wajib. "
             "Decision Tree dan Random Forest tidak membutuhkannya.")
    rentang = (X.max() - X.min())
    st.metric("Disparitas skala (rentang terbesar / terkecil)", f"{rentang.max() / rentang.min():,.0f}x".replace(",", "."))
    fitur = st.selectbox("Bandingkan sebaran fitur", list(X.columns), index=3)
    sc = StandardScaler().fit(Xtr[[fitur]])
    asli, baku = Xtr[fitur], sc.transform(Xtr[[fitur]]).ravel()
    fig = go.Figure()
    fig.add_histogram(x=asli, name="Sebelum scaling", marker_color=WARNA_GANAS, opacity=0.75, nbinsx=35)
    fig.add_histogram(x=baku, name="Sesudah StandardScaler", marker_color=WARNA_JINAK, opacity=0.75, nbinsx=35, xaxis="x2")
    fig.update_layout(xaxis=dict(domain=[0, 0.48], title=f"{fitur} (asli)"), xaxis2=dict(domain=[0.52, 1], title="z-score"), bargap=0.05)
    st.plotly_chart(tata_letak_plotly(fig, 320), width="stretch")
    a, b = st.columns(2)
    a.metric("Mean / SD (asli)", f"{fmt(asli.mean(), 3)} / {fmt(asli.std(), 3)}")
    b.metric("Mean / SD (setelah scaling)", f"{fmt(baku.mean(), 3)} / {fmt(baku.std(ddof=0), 3)}")
    st.caption("StandardScaler hanya menggeser dan menyamakan skala, bentuk sebaran (skewness) tidak berubah.")

with t4:
    st.subheader("Alur modelling")
    a, b, c, d = st.columns(4)
    for kol, (j, ket, det) in zip([a, b, c, d], [
        ("1. Split 80/20", f"Stratified: {len(Xtr)} latih, {len(Xte)} uji", "Kolom id dibuang. Target dikodekan B = 0, M = 1."),
        ("2. Pipeline", "StandardScaler + model", "Scaler dan model di-fit hanya pada data latih di setiap fold."),
        ("3. Stratified 5-Fold CV", "Hanya pada data latih", "Data uji tidak dipakai untuk memilih model."),
        ("4. Uji akhir", f"Sekali pada {len(Xte)} data uji", "random_state = 42 agar hasil dapat diulang."),
    ]):
        with kol:
            with st.container(border=True):
                st.markdown(f"**{j}**")
                st.write(ket)
                st.caption(det)
    st.subheader("Komposisi kelas setelah split")
    kom = pd.DataFrame({"Data latih": ytr.value_counts().sort_index().values,
                        "Data uji": yte.value_counts().sort_index().values}, index=["Jinak (B)", "Ganas (M)"])
    st.dataframe(kom, width="stretch")
    st.caption("Stratified split menjaga proporsi kelas yang sama di data latih dan data uji.")
    st.code(
        "pipeline = Pipeline([\n"
        "    ('imputer', SimpleImputer(strategy='median')),\n"
        "    ('scaler',  StandardScaler()),\n"
        "    ('model',   SVC(kernel='rbf'))\n"
        "])\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)",
        language="python")
