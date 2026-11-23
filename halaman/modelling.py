import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

from utils import (KONFIGURASI, NAMA_MODEL, PEMETAAN_AUDIT, bagi_data, fmt, hasil_cv, hitung_metrik,
                   jalankan_optimasi, kotak, latih_model_dasar, siapkan_xy, skor_model,
                   tata_letak_plotly, vif_dari_korelasi)

st.title("Modelling & Evaluasi")
st.caption("Empat algoritma: SVC RBF, Random Forest, K-NN, dan Decision Tree")

Xtr, Xte, ytr, yte = bagi_data()
X, y = siapkan_xy()
model, tabel_uji, prediksi = latih_model_dasar()
o = jalankan_optimasi()

t1, t2, t3, t4, t5 = st.tabs(["Audit & pemilihan", "Validasi silang", "Metrik evaluasi",
                              "Hasil data uji", "Confusion matrix & ROC"])

# ----------------------------------------------------------------- 1
with t1:
    vif = vif_dari_korelasi(X)
    corr = X.corr().abs().where(np.triu(np.ones((30, 30)), 1).astype(bool)).stack()
    rng = X.max() - X.min()
    a, b, c, d = st.columns(4)
    a.metric("Rasio N/d", f"{fmt(len(X) / 30, 2)} : 1", help="569 sampel, 30 fitur. Sampel relatif kecil, perlu validasi silang")
    b.metric("Fitur dengan VIF > 10", f"{(vif > 10).sum()} dari 30", help="Multikolinearitas ekstrem")
    c.metric("Pasangan korelasi > 0,9", int((corr > 0.9).sum()))
    d.metric("Disparitas skala", f"{rng[rng > 0].max() / rng[rng > 0].min():,.0f}x".replace(",", "."),
             help="StandardScaler wajib untuk SVC dan K-NN")
    st.subheader("Pemetaan temuan audit ke algoritma")
    st.dataframe(PEMETAAN_AUDIT, width="stretch", hide_index=True)
    st.subheader("Konfigurasi 4 model")
    st.dataframe(KONFIGURASI, width="stretch", hide_index=True)
    st.subheader("Cara kerja")
    cols = st.columns(4)
    info = [
        ("SVC RBF", "Mencari batas keputusan non-linear dengan kernel RBF.", "Wajib standardisasi.", "C = 10, gamma = scale"),
        ("Random Forest", "Gabungan banyak pohon yang dilatih pada sampel acak, hasilnya dirata-ratakan.", "Tidak perlu standardisasi.", "100 pohon, max_depth = 10"),
        ("K-NN", "Menentukan kelas dari tetangga terdekat berdasarkan jarak.", "Standardisasi penting karena memakai jarak.", "k = 9, weights = distance"),
        ("Decision Tree", "Membuat aturan percabangan berdasarkan fitur.", "Tidak perlu standardisasi.", "max_depth = 5"),
    ]
    for kol, (nama, cara, perlakuan, konf) in zip(cols, info):
        with kol:
            with st.container(border=True):
                st.markdown(f"**{nama}**")
                st.markdown(f"*Cara kerja:* {cara}")
                st.markdown(f"*Fitur:* {perlakuan}")
                st.markdown(f"*Konfigurasi:* `{konf}`")
    st.caption("Tidak dipakai: MLP / deep learning (sampel terlalu sedikit, rawan overfitting). "
               "SVC RBF juga diuji setelah optimasi (SelectKBest + Grid Search) di halaman Teknik Optimasi.")

# ----------------------------------------------------------------- 2
with t2:
    cv = hasil_cv()
    st.write("Semua model dibungkus pipeline (imputasi median, lalu StandardScaler untuk SVC dan K-NN) dan dievaluasi dengan "
             "**Stratified 5-fold CV** pada data latih.")
    st.dataframe(cv.style.format({c: "{:.4f}" for c in cv.columns if c != "Model"}), width="stretch", hide_index=True)
    urut = cv.sort_values("F1")
    fig = px.bar(urut, x="F1", y="Model", orientation="h", text=urut["F1"].map(lambda v: fmt(v)),
                 color="F1", color_continuous_scale="Tealgrn", title="Peringkat model berdasarkan F1 (CV)")
    fig.update_layout(coloraxis_showscale=False, xaxis=dict(range=[0, 1.05]))
    st.plotly_chart(tata_letak_plotly(fig, 330), width="stretch")

    f = cv.set_index("Model")
    terbaik = cv.iloc[0]["Model"]
    a, b, c = st.columns(3)
    with a:
        kotak(f"<b>{terbaik} teratas</b><br>F1 {fmt(f.loc[terbaik, 'F1'], 3)} dengan simpangan antar-fold paling kecil "
              f"({fmt(f.loc[terbaik, 'F1 (std)'], 3)}).", "sesudah")
    with b:
        kotak(f"<b>Ensemble menolong pohon</b><br>Decision Tree {fmt(f.loc['Decision Tree', 'F1'], 3)} naik menjadi "
              f"{fmt(f.loc['Random Forest', 'F1'], 3)} pada Random Forest.", "info")
    with c:
        kotak(f"<b>Decision Tree tidak stabil</b><br>Simpangan F1 antar-fold {fmt(f.loc['Decision Tree', 'F1 (std)'], 3)}, "
              "paling besar dibanding model lain.", "sebelum")
    st.caption("Selisih antar model teratas bisa lebih kecil daripada simpangan antar-fold, sehingga belum tentu signifikan.")

# ----------------------------------------------------------------- 3
with t3:
    st.subheader("Confusion matrix dan rumus metrik")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(
            "| | Prediksi: Jinak | Prediksi: Ganas |\n|---|---|---|\n"
            "| **Aktual: Jinak** | TN (jinak terprediksi jinak) | FP (jinak terprediksi ganas) |\n"
            "| **Aktual: Ganas** | **FN** (ganas terprediksi jinak) | TP (ganas terprediksi ganas) |")
        st.caption("Kelas positif adalah ganas (M). False negative adalah kesalahan paling berbahaya.")
    with c2:
        st.dataframe(pd.DataFrame([
            ("Accuracy", "(TP + TN) / (TP + TN + FP + FN)", "Proporsi prediksi benar"),
            ("Precision (ganas)", "TP / (TP + FP)", "Dari semua prediksi ganas, berapa yang benar-benar ganas"),
            ("Recall (ganas)", "TP / (TP + FN)", "Dari semua kasus ganas, berapa yang berhasil ditemukan. Prioritas utama"),
            ("F1-Score", "2 x P x R / (P + R)", "Keseimbangan precision dan recall"),
            ("ROC-AUC", "Luas di bawah kurva TPR vs FPR", "Daya pisah skor probabilitas pada semua ambang"),
        ], columns=["Metrik", "Rumus", "Makna"]), width="stretch", hide_index=True)
    kotak("Dalam konteks medis, <b>recall kelas ganas</b> diprioritaskan karena false negative paling berbahaya; "
          "accuracy saja tidak cukup pada data tidak seimbang.", "info", "MENGAPA RECALL?")

# ----------------------------------------------------------------- 4
opt = o["hasil"]["Sesudah"]
kolom_met = ["Accuracy", "Precision (M)", "Recall (M)", "F1 (M)", "ROC-AUC", "FN"]
final = tabel_uji.set_index("Model")[kolom_met].loc[NAMA_MODEL].copy()
final.loc["SVC RBF + optimasi"] = [opt[k] for k in kolom_met]

with t4:
    st.subheader(f"Hasil evaluasi pada data uji ({len(yte)} sampel)")
    st.write("Semua model memakai split 80/20 stratified yang sama dan dievaluasi sekali. "
             "M = ganas (Malignant), FN = false negative (ganas terprediksi jinak).")
    st.dataframe(final.style.format({c: "{:.4f}" for c in final.columns if c != "FN"}).highlight_max(
        subset=["Accuracy", "Precision (M)", "Recall (M)", "F1 (M)", "ROC-AUC"], color="#b7e4c7").highlight_min(
        subset=["FN"], color="#b7e4c7"), width="stretch")
    st.caption("Baris terakhir: SVC RBF setelah SelectKBest (k=25) dan Grid Search (C=10, gamma=0,01).")

    st.subheader("Kestabilan: data uji vs validasi silang")
    k = tabel_uji.set_index("Model").loc[NAMA_MODEL].reset_index()
    k["F1 makro CV"] = k.apply(lambda r: f"{fmt(r['F1 makro CV'])} ± {fmt(r['F1 makro CV std'], 3)}", axis=1)
    st.dataframe(k[["Model", "Accuracy", "Recall (M)", "F1 makro", "ROC-AUC", "F1 makro CV"]].style.format(
        {c: "{:.4f}" for c in ["Accuracy", "Recall (M)", "F1 makro", "ROC-AUC"]}), width="stretch", hide_index=True)
    kk = tabel_uji.set_index("Model")
    a, b, c = st.columns(3)
    with a:
        kotak("<b>Pohon tunggal kurang stabil</b><br>Decision Tree punya simpangan CV terbesar dan recall terendah.", "sebelum")
    with b:
        kotak(f"<b>Waktu latih</b><br>Random Forest {fmt(kk.loc['Random Forest', 'Waktu latih (dtk)'], 3)} dtk, "
              f"SVC {fmt(kk.loc['SVC RBF', 'Waktu latih (dtk)'], 3)} dtk, Decision Tree {fmt(kk.loc['Decision Tree', 'Waktu latih (dtk)'], 3)} dtk.", "info")
    with c:
        kotak(f"<b>Waktu prediksi K-NN</b><br>{fmt(kk.loc['K-NN', 'Waktu prediksi (dtk)'], 3)} dtk karena menghitung jarak ke seluruh data latih.", "info")

# ----------------------------------------------------------------- 5
with t5:
    semua = dict(prediksi)
    semua["SVC RBF + optimasi"] = (o["model"].predict(Xte), skor_model(o["model"], Xte))
    pilihan = st.multiselect("Model yang ditampilkan", list(semua.keys()), default=NAMA_MODEL)
    if not pilihan:
        st.info("Pilih minimal satu model.")
    else:
        cols = st.columns(min(len(pilihan), 3))
        for i, nama in enumerate(pilihan):
            pred, _ = semua[nama]
            m = hitung_metrik(yte, pred)
            cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
            fig = px.imshow(cm, text_auto=True, x=["Jinak", "Ganas"], y=["Jinak", "Ganas"], color_continuous_scale="Blues",
                            labels=dict(x="Prediksi", y="Aktual"), title=nama)
            fig.update_layout(coloraxis_showscale=False)
            with cols[i % len(cols)]:
                st.plotly_chart(tata_letak_plotly(fig, 300), width="stretch")
                st.caption(f"{m['FN']} FN, {m['FP']} FP")
        st.caption("FN = ganas terprediksi jinak (paling berbahaya), FP = jinak terprediksi ganas.")
        fig = go.Figure()
        for nama in pilihan:
            _, skor = semua[nama]
            fpr, tpr, _ = roc_curve(yte, skor)
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{nama} (AUC {fmt(roc_auc_score(yte, skor))})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Acak")
        fig.update_layout(title="Kurva ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(tata_letak_plotly(fig, 440), width="stretch")
        st.caption("ROC-AUC dibaca bersama recall dan confusion matrix. Pilih model berdasarkan recall ganas, bukan akurasi saja.")
