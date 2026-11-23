# Klasifikasi Tumor Payudara WDBC (Streamlit)

Website hasil olah data `data/wdbc.csv` mengikuti isi dua presentasi (EDA, PCA, optimasi, modelling, evaluasi).

## Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```
Buka http://localhost:8501. Pertama kali dibuka, Grid Search (400 pelatihan) berjalan beberapa detik lalu di-cache.

## Struktur
- `app.py` navigasi (st.navigation), `utils.py` data, model, evaluasi, tema
- `halaman/` beranda, dataset, eda, preprocessing, pca, optimasi, modelling, prediksi, massal, kesimpulan
- `data/wdbc.csv` dataset (569 baris, 32 kolom)
