import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import WARNA_GANAS, WARNA_JINAK, fmt, jalankan_optimasi, kotak, tata_letak_plotly

st.title("Teknik Optimasi")
st.caption("Feature Selection dan Hyperparameter Optimization pada SVM")

t1, t2, t3 = st.tabs(["Latar belakang", "Skema & Grid Search", "Hasil sebelum vs sesudah"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Mengapa perlu optimasi?")
        st.markdown(
            "- SVM kernel RBF sangat dipengaruhi nilai **C** dan **gamma**; nilai default belum tentu optimal.\n"
            "- 30 fitur saling berkorelasi (VIF radius_mean sangat tinggi), sehingga banyak fitur redundan dan menambah noise.\n"
            "- Sampel kecil (N/d sekitar 19) membuat model rentan overfitting.\n"
            "- Pada kasus kanker, kasus ganas yang terlewat (false negative) paling berbahaya, sehingga **recall** perlu naik.")
    with c2:
        st.subheader("Teknik yang dipilih")
        kotak("<b>1. Feature Selection</b><br><code>SelectKBest (f_classif)</code> memilih k fitur paling relevan terhadap diagnosis.", "info")
        kotak("<b>2. Hyperparameter Optimization</b><br><b>Grid Search 5-fold</b> mencari kombinasi terbaik k, C, dan gamma.", "info")

o = jalankan_optimasi()

with t2:
    st.subheader("Pipeline")
    a, b, c = st.columns(3)
    a.markdown("**StandardScaler**"); a.caption("Menyamakan skala fitur")
    b.markdown("**SelectKBest**"); b.caption("f_classif, memilih k fitur")
    c.markdown("**SVC (RBF)**"); c.caption("Klasifikasi jinak vs ganas")
    st.subheader("Ruang pencarian Grid Search")
    st.dataframe(pd.DataFrame({
        "Parameter": ["select__k", "svm__C", "svm__gamma"],
        "Nilai": ["10, 15, 20, 25, 30", "0,1 | 1 | 10 | 100", "scale | 0,001 | 0,01 | 0,1"],
        "Jumlah": ["5 nilai", "4 nilai", "4 nilai"]}), width="stretch", hide_index=True)
    kotak("<b>80 kombinasi x 5-fold CV = 400 pelatihan.</b> Split 80/20 stratified (455 latih, 114 uji), "
          "metrik seleksi accuracy, dan data uji hanya dipakai sekali di akhir.", "info")

    tb = o["tabel"].copy()
    tb["k"] = tb["param_seleksi__k"].astype(int)
    tb["C"] = tb["param_model__C"].astype(str)
    tb["gamma"] = tb["param_model__gamma"].astype(str)
    st.subheader("Peta panas akurasi CV")
    k_pilih = st.select_slider("Jumlah fitur terpilih (k)", options=[10, 15, 20, 25, 30], value=int(o["params"]["seleksi__k"]))
    sub = tb[tb["k"] == k_pilih].pivot(index="C", columns="gamma", values="mean_test_score")
    fig = px.imshow(sub, text_auto=".4f", color_continuous_scale="Tealgrn", aspect="auto",
                    labels=dict(x="gamma", y="C", color="Akurasi CV"))
    st.plotly_chart(tata_letak_plotly(fig, 330), width="stretch")
    with st.expander("10 kombinasi terbaik"):
        top = tb.sort_values("mean_test_score", ascending=False).head(10)[["k", "C", "gamma", "mean_test_score", "std_test_score"]]
        top.columns = ["k", "C", "gamma", "Akurasi CV", "Std"]
        st.dataframe(top.style.format({"Akurasi CV": "{:.4f}", "Std": "{:.4f}"}), width="stretch", hide_index=True)

with t3:
    s, t = o["hasil"]["Sebelum"], o["hasil"]["Sesudah"]
    bp = o["params"]
    a, b = st.columns(2)
    a.metric("Parameter terbaik", f"k = {bp['seleksi__k']}, C = {bp['model__C']}, gamma = {bp['model__gamma']}".replace(".", ","))
    b.metric("CV accuracy terbaik", f"{fmt(o['cv_terbaik'], 3)}", help="5-fold, data latih")
    kunci = ["Accuracy", "Precision (M)", "Recall (M)", "F1 (M)", "ROC-AUC"]
    fig = go.Figure()
    fig.add_bar(x=kunci, y=[s[k] for k in kunci], name="Sebelum (30 fitur, C=1)", marker_color="#adb5bd", text=[fmt(s[k]) for k in kunci])
    fig.add_bar(x=kunci, y=[t[k] for k in kunci], name=f"Sesudah (k={bp['seleksi__k']}, C={bp['model__C']}, gamma={bp['model__gamma']})",
                marker_color=WARNA_JINAK, text=[fmt(t[k]) for k in kunci])
    fig.update_layout(barmode="group", yaxis=dict(range=[0.90, 1.005], title="Skor"), title="SVM sebelum vs sesudah optimasi")
    st.plotly_chart(tata_letak_plotly(fig, 400), width="stretch")
    st.caption("Sumbu nilai dimulai dari 0,90.")
    st.metric("False negative", f"{s['FN']}  ->  {t['FN']}  dari {t['FN'] + t['TP']} kasus ganas", delta=f"{t['FN'] - s['FN']}", delta_color="inverse")
    kotak(f"Recall kelas ganas naik dari <b>{fmt(s['Recall (M)'] * 100, 2)}%</b> ke <b>{fmt(t['Recall (M)'] * 100, 2)}%</b>. "
          "Catatan: data uji hanya 114 sampel, sehingga selisih ini setara 2 prediksi.", "sesudah", "CATATAN")
    with st.expander(f"{len(o['fitur_terpilih'])} fitur yang dipilih SelectKBest"):
        st.write(", ".join(o["fitur_terpilih"]))
