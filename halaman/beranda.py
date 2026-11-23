import streamlit as st
from utils import hero, kotak, ringkas_kualitas_data, jalankan_optimasi, fmt

hero("Klasifikasi Tumor Payudara (WDBC)",
     "Optimasi, EDA, PCA, dan Modelling pada dataset Breast Cancer Wisconsin (Diagnostic): jinak (B) vs ganas (M)")

st.caption("Disusun oleh: **Aditia Bagas Hermawan** (241011400553) dan **Adhitya Usman** (241011400558)")

q = ringkas_kualitas_data()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Sampel pasien", q["baris"])
k2.metric("Fitur numerik", 30)
k3.metric("Missing value", q["missing"])
k4.metric("Rasio jinak : ganas", f"{fmt(q['rasio'], 2)} : 1")

st.subheader("Tentang proyek")
st.write(
    "Website ini merangkum seluruh pengerjaan Machine Learning pada dataset WDBC: dari pengenalan data, "
    "pembersihan data, 10 masalah kritis EDA, reduksi dimensi dengan PCA, teknik optimasi (seleksi fitur dan "
    "Grid Search pada SVM), perbandingan empat algoritma (SVC RBF, Random Forest, K-NN, Decision Tree), hingga evaluasi dengan metrik yang sesuai. "
    "Semua angka pada halaman dihitung langsung dari `wdbc.csv`, bukan sekadar disalin."
)
kotak("Dalam kasus kanker, <b>kasus ganas yang terlewat (false negative)</b> jauh lebih berbahaya daripada "
      "jinak yang terprediksi ganas. Karena itu <b>recall kelas ganas</b> menjadi metrik prioritas, "
      "bukan accuracy saja.", "info", "MASALAH UTAMA")

st.subheader("Alur website")
alur = [
    ("1. Dataset", "Profil data WDBC, struktur fitur, dan data mentah"),
    ("2. EDA", "Kualitas data dan 10 masalah kritis beserta mitigasinya"),
    ("3. Preprocessing", "Cleaning, split 80/20 stratified, pipeline tanpa kebocoran"),
    ("4. PCA", "Reduksi 30 fitur menjadi komponen utama dengan 95% varians"),
    ("5. Optimasi", "SelectKBest + Grid Search pada SVC (RBF)"),
    ("6. Modelling", "Empat algoritma, metrik, confusion matrix, dan kurva ROC"),
    ("7. Prediksi", "Coba model untuk satu pasien atau banyak pasien dari CSV"),
    ("8. Kesimpulan", "Temuan, keterbatasan, dan saran lanjutan"),
]
cols = st.columns(4)
for i, (j, d) in enumerate(alur):
    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"**{j}**")
            st.caption(d)

st.subheader("Hasil utama")
o = jalankan_optimasi()
s, t = o["hasil"]["Sebelum"], o["hasil"]["Sesudah"]
a, b, c = st.columns(3)
a.metric("Accuracy SVC setelah optimasi", fmt(t["Accuracy"]), f"{(t['Accuracy']-s['Accuracy'])*100:+.2f} poin")
b.metric("Recall ganas", fmt(t["Recall (M)"]), f"{(t['Recall (M)']-s['Recall (M)'])*100:+.2f} poin")
c.metric("False negative (data uji)", t["FN"], f"{t['FN']-s['FN']} kasus", delta_color="inverse")
st.caption("Pembanding: SVC dengan 30 fitur dan C=1 sebelum optimasi. Data uji hanya 114 sampel, "
           "sehingga selisih ini setara beberapa prediksi saja.")

st.warning("Proyek ini bersifat akademik dan bukan alat diagnosis klinis.", icon=":material/warning:")
