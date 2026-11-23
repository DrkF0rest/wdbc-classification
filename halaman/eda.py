import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import (RANDOM_STATE, WARNA_GANAS, WARNA_JINAK, WARNA_NETRAL, bagi_data, buat_pipa,
                   fmt, kotak, muat_data, ringkas_kualitas_data, siapkan_xy,
                   tata_letak_plotly, vif_dari_korelasi)

st.title("Eksplorasi Data (EDA)")
X, y = siapkan_xy()
q = ringkas_kualitas_data()

# ------------------------------------------------------------------ kualitas data
st.header("Pemeriksaan kualitas data mentah")
st.markdown("**Pemeriksaan bersih**")
a, b, c, d = st.columns(4)
a.metric("Data kosong (missing)", q["missing"])
b.metric("Baris duplikat", q["duplikat"])
c.metric("Anomali fisik", q["anomali"], help="Nilai <= 0 pada radius, perimeter, area, texture")
d.metric("Nilai nol", f"{q['nol_persen']:.2f}%".replace(".", ","), help=f"{q['nol_baris_concavity']} baris concavity = 0, valid secara biologis")
st.markdown("**Temuan yang perlu diperhatikan**")
a, b, c = st.columns(3)
a.metric("Baris mengandung outlier (IQR)", q["outlier_baris"], help=f"{q['outlier_sel']} sel menurut aturan IQR")
b.metric("Fitur sangat miring", f"{q['miring']} dari 30", help="|skewness| > 1")
c.metric("Rasio jinak : ganas", f"{fmt(q['rasio'], 2)} : 1", help="Kelas tidak seimbang ringan")
st.caption("Nilai nol (0,46%) valid secara biologis karena concavity bisa nol pada inti sel yang cembung.")

# ------------------------------------------------------------------ jelajah interaktif
st.header("Jelajah fitur")
tabA, tabB = st.tabs(["Distribusi per diagnosis", "Korelasi antar fitur"])
with tabA:
    fitur = st.selectbox("Pilih fitur", list(X.columns), index=3)
    dd = X[[fitur]].copy()
    dd["Diagnosis"] = np.where(y == 1, "Ganas (M)", "Jinak (B)")
    c1, c2 = st.columns(2)
    f1 = px.histogram(dd, x=fitur, color="Diagnosis", barmode="overlay", opacity=0.65, nbins=40,
                      color_discrete_map={"Jinak (B)": WARNA_JINAK, "Ganas (M)": WARNA_GANAS})
    c1.plotly_chart(tata_letak_plotly(f1, 340), width="stretch")
    f2 = px.box(dd, x="Diagnosis", y=fitur, color="Diagnosis",
                color_discrete_map={"Jinak (B)": WARNA_JINAK, "Ganas (M)": WARNA_GANAS})
    c2.plotly_chart(tata_letak_plotly(f2, 340), width="stretch")
    st.caption(f"Skewness {fitur}: {fmt(X[fitur].skew(), 3)}")
with tabB:
    ukuran = st.radio("Cakupan", ["10 fitur mean", "Semua 30 fitur"], horizontal=True)
    kolom = [c for c in X.columns if c.endswith("_mean")] if ukuran.startswith("10") else list(X.columns)
    cm = X[kolom].corr()
    fig = px.imshow(cm, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(tata_letak_plotly(fig, 560 if len(kolom) > 10 else 460), width="stretch")
    pas = (cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack().abs() > 0.9).sum()
    st.caption(f"Pada cakupan ini ada {pas} pasangan fitur dengan |korelasi| > 0,9 (radius, perimeter, dan area hampir identik).")

# ------------------------------------------------------------------ 10 masalah
st.header("10 masalah kritis EDA")
st.write("Setiap masalah didiagnosis, dimitigasi, lalu dibuktikan dengan perbandingan **sebelum vs sesudah**. "
         "Angka dihitung langsung dari data.")


def sebelum_sesudah(sebelum, sesudah):
    c1, c2 = st.columns(2)
    with c1:
        kotak(sebelum, "sebelum", "SEBELUM")
    with c2:
        kotak(sesudah, "sesudah", "SESUDAH")


@st.cache_data
def hitung_masalah():
    r = {}
    Xtr, Xte, ytr, yte = bagi_data()
    # 1 kebocoran data
    Xs = StandardScaler().fit_transform(X)
    a, b, c, d = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    r["acc_bocor"] = accuracy_score(d, SVC(C=10, random_state=RANDOM_STATE).fit(a, c).predict(b))
    r["acc_pipa"] = accuracy_score(yte, buat_pipa(SVC(C=10, random_state=RANDOM_STATE)).fit(Xtr, ytr).predict(Xte))
    # 2 skewness
    r["skew_raw"] = X["area_mean"].skew()
    r["skew_std"] = pd.Series(StandardScaler().fit_transform(X[["area_mean"]]).ravel()).skew()
    r["skew_log"] = np.log1p(X["area_mean"]).skew()
    # 3 listwise deletion (simulasi missing 61 baris berkaitan dengan target)
    rng = np.random.default_rng(RANDOM_STATE)
    w = np.where(y == 1, 3.0, 1.0)
    idx = rng.choice(len(X), 61, replace=False, p=w / w.sum())
    Xm = X.copy()
    Xm.iloc[idx, Xm.columns.get_loc("area_mean")] = np.nan
    ok = Xm.notna().all(axis=1)
    r["n_drop"] = int(ok.sum())
    r["geser"] = y[ok].mean() - y.mean()
    Xsc = pd.DataFrame(StandardScaler().fit_transform(Xm), columns=Xm.columns)
    r["n_knn"] = KNNImputer(n_neighbors=5, add_indicator=True).fit_transform(Xsc).shape[0]
    # 4 multikolinearitas
    f4 = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]
    r["vif_sebelum"] = vif_dari_korelasi(X[f4])
    r["vif_sesudah"] = vif_dari_korelasi(X[["radius_mean", "texture_mean"]])
    # 5 paradoks akurasi
    dm = np.zeros(len(y), dtype=int)   # selalu menebak kelas mayoritas (jinak)
    r["dummy_acc"], r["dummy_f1"] = accuracy_score(y, dm), f1_score(y, dm, average="macro")
    lr = make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE))
    f2 = ["radius_mean", "texture_mean"]
    lr.fit(X[f2], y)
    r["lr_f1"] = f1_score(y, lr.predict(X[f2]), average="macro")
    r["lr_auc"] = roc_auc_score(y, lr.predict_proba(X[f2])[:, 1])
    # 6 korelasi
    f3 = ["radius_mean", "area_mean", "texture_mean"]
    r["pearson"] = [stats.pearsonr(X[c], y)[0] for c in f3]
    r["spearman"] = [stats.spearmanr(X[c], y)[0] for c in f3]
    r["mi"] = mutual_info_classif(X[f3], y, random_state=RANDOM_STATE)
    # 7 fitur ID
    df = muat_data()
    kol = ["id", "radius_mean", "texture_mean"]
    r["imp_id"] = dict(zip(kol, DecisionTreeClassifier(random_state=RANDOM_STATE).fit(df[kol], y).feature_importances_))
    r["imp_noid"] = dict(zip(kol[1:], DecisionTreeClassifier(random_state=RANDOM_STATE).fit(df[kol[1:]], y).feature_importances_))
    r["unik_id"] = df["id"].nunique() / len(df)
    # 8 masking effect
    s = pd.concat([X["area_mean"], pd.Series([5000.0, 6000.0])], ignore_index=True)
    z = ((s - s.mean()) / s.std()).abs()
    med = s.median(); mad = (s - med).abs().median()
    mz = (0.6745 * (s - med) / mad).abs()
    r["z_klasik"], r["z_mod"] = int((z > 3).sum()), int((mz > 3.5).sum())
    # 9 multimodalitas
    am = X["area_mean"]
    r["mean_all"], r["mean_b"], r["mean_m"] = am.mean(), am[y == 0].mean(), am[y == 1].mean()
    r["p_uji"] = stats.ttest_ind(am[y == 0], am[y == 1], equal_var=False).pvalue
    # 10 outlier valid
    aw = X["area_worst"]
    q1, q3 = aw.quantile([0.25, 0.75])
    out = aw > q3 + 1.5 * (q3 - q1)
    r["n_out"], r["out_ganas"], r["batas"] = int(out.sum()), y[out].mean(), q3 + 1.5 * (q3 - q1)
    r["skew_aw"] = aw.skew()
    r["skew_aw_yj"] = pd.Series(PowerTransformer("yeo-johnson").fit_transform(X[["area_worst"]]).ravel()).skew()
    return r


R = hitung_masalah()
ganas_persen = f"{y.mean() * 100:.1f}%".replace(".", ",")
geser_txt = f"{R['geser']:+.3f}".replace(".", ",")

g1, g2, g3 = st.tabs(["Masalah 1-4: Kualitas data & preprocessing",
                      "Masalah 5-7: Metrik, korelasi, identitas",
                      "Masalah 8-10: Outlier & subpopulasi"])

with g1:
    st.subheader("1. Kebocoran data (data leakage)")
    sebelum_sesudah(
        f"Scaler di-fit <b>sebelum</b> train-test split, sehingga informasi data uji ikut masuk.<br>Akurasi uji: <b>{fmt(R['acc_bocor'])}</b> (preprocessing bocor).",
        f"Semua langkah dibungkus <b>Pipeline</b>, scaler hanya di-fit pada data latih.<br>Akurasi uji: <b>{fmt(R['acc_pipa'])}</b>. Tidak ada selisih pada split ini, tetapi isolasi tetap wajib.")

    st.subheader("2. Skewness ekstrem pada area_mean")
    sebelum_sesudah(
        f"StandardScaler tidak mengubah bentuk sebaran.<br>Skewness setelah StandardScaler: <b>{fmt(R['skew_std'], 3)}</b> (asli {fmt(R['skew_raw'], 3)}).",
        f"Transformasi <b>log1p</b> memampatkan ekor kanan.<br>Skewness turun ke <b>{fmt(R['skew_log'], 3)}</b>.")
    fig = go.Figure()
    fig.add_histogram(x=X["area_mean"], name="Asli", marker_color=WARNA_GANAS, opacity=0.7, nbinsx=40)
    fig.add_histogram(x=np.log1p(X["area_mean"]), name="log1p", marker_color=WARNA_JINAK, opacity=0.7, nbinsx=40, xaxis="x2")
    fig.update_layout(xaxis=dict(title="area_mean (asli)", domain=[0, 0.48]),
                      xaxis2=dict(title="log1p(area_mean)", domain=[0.52, 1]), bargap=0.05)
    st.plotly_chart(tata_letak_plotly(fig, 300), width="stretch")

    st.subheader("3. Listwise deletion")
    sebelum_sesudah(
        f"<code>dropna()</code> pada missing yang terkait target (simulasi 61 baris). Data tersisa <b>{R['n_drop']}</b> baris, rasio ganas bergeser <b>{geser_txt}</b> (distorsi).",
        f"<b>KNNImputer + indikator missing</b>: <b>{R['n_knn']}</b> baris ({R['n_knn'] / len(X) * 100:.0f}%) terjaga.")

    st.subheader("4. Multikolinearitas tinggi")
    v1, v2 = R["vif_sebelum"], R["vif_sesudah"]
    sebelum_sesudah(
        f"radius, perimeter, dan area berkorelasi hampir sempurna.<br>VIF: radius <b>{fmt(v1['radius_mean'], 1)}</b>; perimeter <b>{fmt(v1['perimeter_mean'], 1)}</b>; area <b>{fmt(v1['area_mean'], 1)}</b>.",
        f"Setelah drop perimeter dan area, VIF turun menjadi <b>{fmt(v2.max(), 2)}</b> (sekitar 1).")
    fv = px.bar(v1.reset_index().rename(columns={"index": "Fitur", 0: "VIF"}), x="Fitur", y="VIF",
                color_discrete_sequence=[WARNA_GANAS], title="VIF (radius, texture, perimeter, area)")
    fv.add_hline(y=10, line_dash="dash", annotation_text="batas VIF = 10")
    st.plotly_chart(tata_letak_plotly(fv, 300), width="stretch")

with g2:
    st.subheader("5. Paradoks akurasi")
    sebelum_sesudah(
        f"Akurasi menyesatkan pada kelas tidak seimbang (jinak 62,7% : ganas {ganas_persen}).<br>Dummy mayoritas: akurasi <b>{fmt(R['dummy_acc'] * 100, 1)}%</b>, tetapi F1 makro hanya <b>{fmt(R['dummy_f1'], 3)}</b>.",
        f"<code>class_weight='balanced'</code> (SVC RBF pada radius_mean dan texture_mean, dievaluasi pada seluruh data): F1 makro <b>{fmt(R['lr_f1'], 3)}</b> dan ROC-AUC <b>{fmt(R['lr_auc'], 3)}</b>.")

    st.subheader("6. Bias korelasi Pearson")
    st.write("Pearson hanya mengukur hubungan **linear**. Pembandingnya Spearman (monotonik) dan Mutual Information (non-linear).")
    tab_k = pd.DataFrame({"Fitur": ["radius_mean", "area_mean", "texture_mean"],
                          "Pearson": R["pearson"], "Spearman": R["spearman"], "Mutual Info": R["mi"]}).set_index("Fitur")
    st.dataframe(tab_k.style.format("{:.3f}"), width="stretch")
    kotak("Peringkat fitur konsisten di ketiga ukuran: radius dan area jauh lebih informatif daripada texture. "
          "Pilih metrik korelasi yang sesuai dengan bentuk hubungan.", "sesudah", "TEMUAN")

    st.subheader("7. Fitur ID berkardinalitas tinggi")
    sebelum_sesudah(
        f"<code>id</code> memiliki rasio keunikan <b>{fmt(R['unik_id'], 2)}</b>. Pohon keputusan memberi bobot <b>{fmt(R['imp_id']['id'], 2)}</b> pada nomor ID.",
        f"ID dibuang: bobot kembali ke radius_mean <b>{fmt(R['imp_noid']['radius_mean'], 2)}</b> dan texture_mean <b>{fmt(R['imp_noid']['texture_mean'], 2)}</b>.")
    st.info("Pelajaran: pilih metrik dan fitur dengan diagnosis yang tepat, jangan hanya mengandalkan akurasi atau korelasi linear.")

with g3:
    st.subheader("8. Masking effect outlier")
    sebelum_sesudah(
        f"Mean dan SD terinflasi oleh outlier ekstrem (5000 dan 6000 ditambahkan ke area_mean).<br>Z-Score klasik mendeteksi hanya <b>{R['z_klasik']}</b> data.",
        f"Modified Z-Score (median dan MAD) tidak ter-masking.<br>Mendeteksi <b>{R['z_mod']}</b> data.")

    st.subheader("9. Multimodalitas fitur")
    sebelum_sesudah(
        f"Mean global area_mean: <b>{fmt(R['mean_all'], 2)}</b>. Mean global menyembunyikan dua subpopulasi.",
        f"Jinak <b>{fmt(R['mean_b'], 2)}</b> vs ganas <b>{fmt(R['mean_m'], 2)}</b> (uji-t signifikan, p &lt; 0,001).")
    am = pd.DataFrame({"area_mean": X["area_mean"], "Diagnosis": np.where(y == 1, "Ganas (M)", "Jinak (B)")})
    fh = px.histogram(am, x="area_mean", color="Diagnosis", barmode="overlay", opacity=0.65, nbins=40,
                      color_discrete_map={"Jinak (B)": WARNA_JINAK, "Ganas (M)": WARNA_GANAS})
    fh.add_vline(x=R["mean_all"], line_dash="dash", line_color=WARNA_NETRAL, annotation_text="mean global")
    st.plotly_chart(tata_letak_plotly(fh, 320), width="stretch")

    st.subheader("10. Menghapus outlier yang valid")
    sebelum_sesudah(
        f"Outlier IQR pada area_worst (batas atas {fmt(R['batas'], 1)}) justru kasus penting: <b>{R['out_ganas'] * 100:.0f}%</b> dari {R['n_out']} data di atas pagar IQR adalah ganas (populasi: {ganas_persen}).",
        f"Data <b>dipertahankan</b> ({len(X)} baris); skala dikompresi dengan Yeo-Johnson. Skewness area_worst {fmt(R['skew_aw'], 2)} menjadi {fmt(R['skew_aw_yj'], 2)}.")
    aw = pd.DataFrame({"area_worst": X["area_worst"], "Diagnosis": np.where(y == 1, "Ganas (M)", "Jinak (B)")})
    fb = px.box(aw, x="Diagnosis", y="area_worst", color="Diagnosis", points="outliers",
                color_discrete_map={"Jinak (B)": WARNA_JINAK, "Ganas (M)": WARNA_GANAS})
    st.plotly_chart(tata_letak_plotly(fb, 320), width="stretch")
