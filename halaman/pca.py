import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (RANDOM_STATE, WARNA_GANAS, WARNA_JINAK, bagi_data, fmt, kotak,
                   tata_letak_plotly)

st.title("Reduksi Dimensi (PCA)")
st.caption("Dari 30 fitur menjadi komponen utama yang menjaga 95% varians")

Xtr, Xte, ytr, yte = bagi_data()

c1, c2, c3 = st.columns(3)
with c1:
    kotak("<b>Scaling wajib</b><br>StandardScaler sebelum PCA karena PCA sensitif terhadap skala.", "info")
with c2:
    kotak("<b>Target 95% varians</b><br><code>n_components = 0,95</code>, bukan menebak jumlah kolom.", "info")
with c3:
    kotak("<b>Tanpa leakage</b><br>Scaler, PCA, dan model berada dalam satu Pipeline.", "info")

target = st.slider("Target varians yang dipertahankan", 0.80, 0.99, 0.95, 0.01,
                   help="Default tugas: 95%")
scaler = StandardScaler().fit(Xtr)
Z = scaler.transform(Xtr)
pca_penuh = PCA(random_state=RANDOM_STATE).fit(Z)
kum = np.cumsum(pca_penuh.explained_variance_ratio_)
n_komp = int(np.searchsorted(kum, target) + 1)

a, b, c = st.columns(3)
a.metric("Fitur asli", 30)
b.metric("Komponen utama", n_komp, f"-{(1 - n_komp / 30) * 100:.0f}% dimensi", delta_color="off")
c.metric("Varians terjaga", f"{kum[n_komp - 1] * 100:.2f}%".replace(".", ","))

col_l, col_r = st.columns(2)
with col_l:
    fig = go.Figure()
    fig.add_bar(x=list(range(1, 31)), y=pca_penuh.explained_variance_ratio_, name="Varians per komponen",
                marker_color="#adb5bd")
    fig.add_scatter(x=list(range(1, 31)), y=kum, name="Varians kumulatif", mode="lines+markers",
                    line=dict(color=WARNA_GANAS))
    fig.add_hline(y=target, line_dash="dash", annotation_text=f"{target * 100:.0f}%")
    fig.add_vline(x=n_komp, line_dash="dot")
    fig.update_layout(title="Scree plot dan varians kumulatif", xaxis_title="Komponen", yaxis_title="Proporsi varians")
    st.plotly_chart(tata_letak_plotly(fig, 380), width="stretch")
with col_r:
    p2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z)
    P = p2.transform(Z)
    dd = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "Diagnosis": np.where(ytr.values == 1, "Ganas (M)", "Jinak (B)")})
    f2 = px.scatter(dd, x="PC1", y="PC2", color="Diagnosis", opacity=0.75,
                    color_discrete_map={"Jinak (B)": WARNA_JINAK, "Ganas (M)": WARNA_GANAS},
                    title=f"Proyeksi 2D data latih ({p2.explained_variance_ratio_.sum() * 100:.1f}% varians)".replace(".", ",", 1))
    st.plotly_chart(tata_letak_plotly(f2, 380), width="stretch")
st.caption("Proyeksi 2D hanya menjaga sebagian varians, namun kelas masih terlihat terpisah.")

st.header("Dampak PCA terhadap performa dan efisiensi")
st.write(f"SVC RBF dilatih pada **30 fitur (baseline)** dan pada **{n_komp} komponen PCA**. "
         "Waktu diukur langsung di mesin ini sehingga dapat berbeda dari slide.")


@st.cache_data
def bandingkan(n):
    hasil = {}
    for nama, langkah in [("Baseline", [("scaler", StandardScaler())]),
                          ("PCA", [("scaler", StandardScaler()), ("pca", PCA(n_components=n, random_state=RANDOM_STATE))])]:
        m = Pipeline(langkah + [("model", SVC(C=10, probability=True, random_state=RANDOM_STATE))])
        t0 = time.perf_counter(); m.fit(Xtr, ytr); tl = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = m.predict(Xte); ti = time.perf_counter() - t0
        hasil[nama] = dict(acc=accuracy_score(yte, pred), f1=f1_score(yte, pred),
                           auc=roc_auc_score(yte, m.predict_proba(Xte)[:, 1]), tl=tl, ti=ti)
    return hasil


h = bandingkan(n_komp)
bm, pc = h["Baseline"], h["PCA"]
tabel = pd.DataFrame({
    "Metrik": ["Jumlah fitur", "Varians terjaga", "Waktu latih (detik)", "Waktu inferensi (detik)",
               "Akurasi uji", "F1-Score (ganas)", "ROC-AUC"],
    "Baseline (30 fitur)": ["30", "100%", fmt(bm["tl"], 3), fmt(bm["ti"], 3), fmt(bm["acc"]), fmt(bm["f1"]), fmt(bm["auc"])],
    f"PCA ({n_komp} komponen)": [str(n_komp), f"{kum[n_komp - 1] * 100:.2f}%".replace(".", ","), fmt(pc["tl"], 3),
                                 fmt(pc["ti"], 3), fmt(pc["acc"]), fmt(pc["f1"]), fmt(pc["auc"])],
})
st.dataframe(tabel, width="stretch", hide_index=True)

a, b, c = st.columns(3)
with a:
    kotak(f"<b>Dimensi turun {(1 - n_komp / 30) * 100:.0f}%</b><br>Akurasi uji {fmt(bm['acc'])} menjadi {fmt(pc['acc'])}, F1 {fmt(bm['f1'])} menjadi {fmt(pc['f1'])} (selisih kecil pada data uji 114 sampel).", "sesudah")
with b:
    if pc["tl"] < bm["tl"]:
        kotak(f"<b>Waktu latih sedikit turun</b><br>{fmt(bm['tl'], 3)} dtk menjadi {fmt(pc['tl'], 3)} dtk. Selisihnya sangat kecil dan bergantung mesin; manfaat kecepatan baru nyata pada data besar.", "info")
    else:
        kotak("<b>Belum lebih cepat</b><br>Waktu latih naik karena overhead PCA pada data sekecil ini. Manfaat kecepatan baru terasa pada data besar.", "info")
with c:
    kotak("<b>Komponen ortogonal</b><br>Multikolinearitas (masalah #4) hilang secara matematis.", "sesudah")

with st.expander("Kontribusi fitur pada komponen utama (loading)"):
    pca_t = PCA(n_components=n_komp, random_state=RANDOM_STATE).fit(Z)
    muat = pd.DataFrame(pca_t.components_.T, index=Xtr.columns,
                        columns=[f"PC{i + 1}" for i in range(n_komp)])
    st.dataframe(muat.iloc[:, :5].style.format("{:.3f}").background_gradient(cmap="RdBu_r", axis=None),
                 width="stretch")
    st.caption("Lima komponen pertama. PC1 didominasi ciri ukuran dan bentuk (radius, perimeter, area, concavity).")
