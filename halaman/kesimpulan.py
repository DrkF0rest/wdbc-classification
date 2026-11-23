import streamlit as st

from utils import fmt, jalankan_optimasi, kotak

st.title("Kesimpulan & Saran")

o = jalankan_optimasi()
s, t = o["hasil"]["Sebelum"], o["hasil"]["Sesudah"]

st.header("Kesimpulan")
c1, c2 = st.columns(2)
with c1:
    kotak(f"<b>1. Optimasi efektif</b><br>SelectKBest + Grid Search menaikkan akurasi SVM dari {fmt(s['Accuracy'] * 100, 2)}% ke "
          f"{fmt(t['Accuracy'] * 100, 2)}% dan recall ganas dari {fmt(s['Recall (M)'] * 100, 2)}% ke {fmt(t['Recall (M)'] * 100, 2)}%.", "sesudah")
    kotak("<b>3. PCA menyederhanakan</b><br>30 fitur menjadi 10 komponen (95,21% varians) dengan performa mendekati baseline, "
          "tetapi belum mempercepat pelatihan pada data sekecil ini.", "sesudah")
with c2:
    kotak("<b>2. 10 masalah EDA terbukti</b><br>Dampak terbesar: multikolinearitas (VIF sangat tinggi), outlier valid "
          "(100% ganas), dan ID yang menyerap importance.", "sesudah")
    kotak("<b>4. Metrik menentukan model</b><br>SVC RBF teratas pada validasi silang, disusul Random Forest dan K-NN, sementara Decision Tree tunggal paling lemah; "
          "accuracy saja tidak cukup, recall kelas ganas jadi prioritas.", "sesudah")
kotak("<b>Data bersih dan siap dimodelkan</b><br>569 baris, 0 missing, 0 duplikat, 0 anomali fisik. Cleaning cukup membuang id, "
      "mengodekan target, dan scaling di dalam pipeline.", "info")

st.header("Keterbatasan dan saran lanjutan")
a, b = st.columns(2)
with a:
    st.subheader("Keterbatasan")
    st.markdown(
        "- Data uji hanya 114 sampel; satu kasus mengubah recall sekitar 2,4 poin.\n"
        "- Evaluasi memakai satu kali split, sehingga hasil bisa berubah pada pembagian lain.\n"
        "- Selisih antar model teratas lebih kecil dari simpangan antar-fold.\n"
        "- Hasil bersifat akademik dan bukan alat diagnosis klinis.")
with b:
    st.subheader("Saran")
    st.markdown(
        "- Gunakan repeated atau nested cross-validation untuk estimasi yang lebih stabil.\n"
        "- Turunkan ambang probabilitas (di bawah 0,5) untuk menaikkan recall ganas.\n"
        "- Optimasi hyperparameter semua model, bukan hanya SVM.\n"
        "- Uji kombinasi PCA + SVM pada data yang lebih besar.")
st.warning("Website ini untuk keperluan pembelajaran dan bukan pengganti diagnosis tenaga medis.", icon=":material/warning:")
