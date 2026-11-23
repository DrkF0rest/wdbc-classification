import pandas as pd
import plotly.express as px
import streamlit as st
from utils import (CIRI_INTI, STATISTIK, WARNA_GANAS, WARNA_JINAK, fmt, kotak,
                   muat_data, ringkas_kualitas_data, tata_letak_plotly)

st.title("Tentang Dataset")
st.caption("Breast Cancer Wisconsin (Diagnostic), WDBC")

df = muat_data()
q = ringkas_kualitas_data()

t1, t2, t3, t4 = st.tabs(["Latar belakang", "Struktur fitur", "Data mentah", "Distribusi diagnosis"])

with t1:
    c1, c2, c3 = st.columns(3)
    with c1:
        kotak("Dataset berisi pengukuran inti sel dari citra biopsi jarum halus (FNA) massa payudara. "
              "Ciri seperti ukuran, tekstur, dan kecekungan inti sel membedakan tumor jinak dan ganas.",
              "info", "LATAR BELAKANG")
    with c2:
        kotak("Menentukan apakah tumor jinak (B) atau ganas (M) dari 30 ciri numerik. Kesalahan tidak setara: "
              "kasus ganas yang terprediksi jinak (false negative) jauh lebih berbahaya daripada sebaliknya.",
              "info", "MASALAH")
    with c3:
        kotak("Membandingkan algoritma klasifikasi pada data yang sudah bersih, lalu memilih yang terbaik dengan "
              "metrik yang sesuai: recall kelas ganas, F1-Score, dan ROC-AUC.", "info", "TUJUAN")
    st.markdown("**Pertanyaan utama:** algoritma mana yang paling andal menemukan kasus ganas pada data yang sudah bersih?")

with t2:
    st.write("Ada 10 ciri inti sel yang masing-masing dihitung dalam tiga statistik: "
             "**mean** (rata-rata), **se** (standard error), dan **worst** (nilai terburuk/terbesar). "
             "Hasilnya 10 x 3 = 30 fitur numerik.")
    tabel = pd.DataFrame({s: [f"{c}_{s}" for c in CIRI_INTI] for s in STATISTIK}, index=CIRI_INTI)
    tabel.index.name = "Ciri inti sel"
    st.dataframe(tabel, width="stretch")
    st.caption("Kolom lain: `id` (pengenal pasien) dan `diagnosis` (target: B = jinak, M = ganas).")

with t3:
    a, b, c = st.columns(3)
    a.metric("Baris (sampel)", q["baris"])
    b.metric("Kolom", q["kolom"])
    c.metric("Sel nilai fitur", f"{q['baris'] * 30:,}".replace(",", "."))
    st.markdown("**Kolom:** `id` (pengenal pasien), `diagnosis` (target B/M), dan 30 fitur = 10 ciri x (mean, se, worst).")
    pilih = st.radio("Tampilkan", ["Semua", "Jinak (B)", "Ganas (M)"], horizontal=True)
    n = st.slider("Jumlah baris", 5, 50, 5)
    tampil = df if pilih == "Semua" else df[df["diagnosis"] == ("B" if "Jinak" in pilih else "M")]
    st.dataframe(tampil.head(n), width="stretch", hide_index=True)
    st.caption("Tipe data: 30 float, 1 integer (id), dan 1 teks (diagnosis).")
    with st.expander("Tipe data tiap kolom"):
        st.dataframe(df.dtypes.astype(str).rename("Tipe").to_frame(), width="stretch")
    with st.expander("Statistik deskriptif"):
        st.dataframe(df.drop(columns="id").describe().T, width="stretch")

with t4:
    cA, cB = st.columns([1, 1])
    dist = pd.DataFrame({"Diagnosis": [f"Jinak (B): {q['jinak']}", f"Ganas (M): {q['ganas']}"],
                         "Jumlah": [q["jinak"], q["ganas"]]})
    fig = px.pie(dist, names="Diagnosis", values="Jumlah", hole=0.5,
                 color="Diagnosis", color_discrete_sequence=[WARNA_JINAK, WARNA_GANAS])
    cA.plotly_chart(tata_letak_plotly(fig, 340), width="stretch")
    with cB:
        st.metric("Jinak", f"{q['jinak']} ({q['jinak'] / q['baris'] * 100:.1f}%)".replace(".", ","))
        st.metric("Ganas", f"{q['ganas']} ({q['ganas'] / q['baris'] * 100:.1f}%)".replace(".", ","))
        st.metric("Rasio jinak : ganas", f"{fmt(q['rasio'], 2)} : 1")
    st.caption("Kelas tidak seimbang ringan: ditangani dengan metrik yang sesuai dan stratified split, tanpa SMOTE.")
