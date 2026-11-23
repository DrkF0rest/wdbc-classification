"""
Fungsi bersama untuk website Klasifikasi Tumor Payudara WDBC (Streamlit).
Berisi: tema, pemuatan data, definisi model, evaluasi, dan cache hasil pelatihan.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "wdbc.csv"
RANDOM_STATE = 42

WARNA_JINAK = "#2a9d8f"
WARNA_GANAS = "#e63946"
WARNA_AKSEN = "#d6336c"
WARNA_NETRAL = "#6c757d"

CIRI_INTI = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
             "concavity", "concave_points", "symmetry", "fractal_dimension"]
STATISTIK = ["mean", "se", "worst"]


# --------------------------------------------------------------------------- #
# Tema
# --------------------------------------------------------------------------- #
def pasang_tema():
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}
        h1 {{ letter-spacing: -0.5px; }}
        [data-testid="stMetric"] {{
            background: rgba(128,128,128,0.08);
            border: 1px solid rgba(128,128,128,0.20);
            border-left: 5px solid {WARNA_AKSEN};
            padding: 0.8rem 1rem; border-radius: 10px;
        }}
        .kotak {{
            border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.5rem 0 1rem 0;
            border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.07);
        }}
        .kotak.sebelum {{ border-left: 5px solid {WARNA_GANAS}; }}
        .kotak.sesudah {{ border-left: 5px solid {WARNA_JINAK}; }}
        .kotak.info    {{ border-left: 5px solid {WARNA_AKSEN}; }}
        .kotak b.tag   {{ font-size: 0.75rem; letter-spacing: 1px; }}
        .hero {{
            padding: 2rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
            background: linear-gradient(135deg, {WARNA_AKSEN} 0%, #7b2cbf 100%); color: white;
        }}
        .hero h1 {{ color: white; margin: 0 0 .3rem 0; }}
        .hero p  {{ color: rgba(255,255,255,.92); margin: 0; font-size: 1.05rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(judul, subjudul):
    st.markdown(f'<div class="hero"><h1>{judul}</h1><p>{subjudul}</p></div>', unsafe_allow_html=True)


def kotak(isi, jenis="info", tag=None):
    label = f'<b class="tag">{tag}</b><br>' if tag else ""
    st.markdown(f'<div class="kotak {jenis}">{label}{isi}</div>', unsafe_allow_html=True)


def tata_letak_plotly(fig, tinggi=380):
    fig.update_layout(height=tinggi, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def fmt(x, d=4):
    """Format angka gaya Indonesia (koma desimal)."""
    return f"{x:.{d}f}".replace(".", ",")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data
def muat_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


@st.cache_data
def siapkan_xy():
    df = muat_data()
    y = (df["diagnosis"] == "M").astype(int)      # M (ganas) = 1 = kelas positif
    X = df.drop(columns=["id", "diagnosis"])
    return X, y


@st.cache_data
def bagi_data():
    X, y = siapkan_xy()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def daftar_fitur():
    X, _ = siapkan_xy()
    return list(X.columns)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _pipe(model, scale=True):
    langkah = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        langkah.append(("scaler", StandardScaler()))
    langkah.append(("model", model))
    return Pipeline(langkah)


NAMA_MODEL = ["SVC RBF", "Random Forest", "K-NN", "Decision Tree"]


def model_empat():
    """Empat algoritma yang dipakai: SVC RBF, Random Forest, K-NN, Decision Tree."""
    return {
        "SVC RBF": _pipe(SVC(C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)),
        "Random Forest": _pipe(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE), False),
        "K-NN": _pipe(KNeighborsClassifier(n_neighbors=9, weights="distance")),
        "Decision Tree": _pipe(DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), False),
    }


KONFIGURASI = pd.DataFrame(
    [
        ("Non-parametrik", "SVC kernel RBF", "C = 10, gamma = scale", "Ya (StandardScaler)"),
        ("Pohon dan ensemble", "Random Forest", "100 pohon, max_depth = 10", "Tidak perlu"),
        ("Non-parametrik", "K-NN Classifier", "k = 9, weights = distance", "Ya (StandardScaler)"),
        ("Pohon dan ensemble", "Decision Tree", "max_depth = 5", "Tidak perlu"),
    ],
    columns=["Kelompok", "Model", "Konfigurasi", "Perlu scaling"],
)

PEMETAAN_AUDIT = pd.DataFrame(
    [
        ("Skewness dan hubungan non-linear", "SVC kernel RBF", "Batas keputusan non-linear"),
        ("Skala berbeda jauh, pola lokal", "K-NN (weights distance)", "Efektif setelah StandardScaler"),
        ("Pencilan dan kolinearitas (VIF > 10)", "Random Forest", "Invarian skala, tahan pencilan dan fitur berkorelasi"),
        ("Pembanding sederhana, mudah dibaca", "Decision Tree", "Aturan percabangan yang transparan; batas bawah untuk ensemble"),
    ],
    columns=["Temuan audit", "Algoritma terpilih", "Alasan"],
)


def model_svc_optimasi(k=25, C=10, gamma=0.01, probabilitas=True):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("seleksi", SelectKBest(f_classif, k=k)),
        ("model", SVC(C=C, gamma=gamma, probability=probabilitas, random_state=RANDOM_STATE)),
    ])


# --------------------------------------------------------------------------- #
# Evaluasi
# --------------------------------------------------------------------------- #
def skor_model(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.decision_function(X)


def hitung_metrik(y_true, y_pred, skor=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    hasil = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (M)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (M)": recall_score(y_true, y_pred, zero_division=0),
        "F1 (M)": f1_score(y_true, y_pred, zero_division=0),
        "F1 makro": f1_score(y_true, y_pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_true, skor) if skor is not None else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    return hasil


@st.cache_resource(show_spinner="Melatih model dasar...")
def latih_model_dasar():
    """Latih 4 model pada data latih, evaluasi sekali pada data uji."""
    Xtr, Xte, ytr, yte = bagi_data()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    model, baris, prediksi = {}, [], {}
    for nama, m in model_empat().items():
        t0 = time.perf_counter(); m.fit(Xtr, ytr); t_latih = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = m.predict(Xte); t_pred = time.perf_counter() - t0
        skor = skor_model(m, Xte)
        met = hitung_metrik(yte, pred, skor)
        cvs = cross_validate(m, Xtr, ytr, cv=cv, scoring="f1_macro")["test_score"]
        met.update({"Model": nama, "F1 makro CV": cvs.mean(), "F1 makro CV std": cvs.std(),
                    "Waktu latih (dtk)": t_latih, "Waktu prediksi (dtk)": t_pred})
        baris.append(met); model[nama] = m; prediksi[nama] = (pred, skor)
    return model, pd.DataFrame(baris), prediksi


@st.cache_resource(show_spinner="Menjalankan Grid Search (80 kombinasi x 5-fold)...")
def jalankan_optimasi():
    """Pipeline StandardScaler -> SelectKBest -> SVC (RBF) dengan Grid Search 5-fold."""
    Xtr, Xte, ytr, yte = bagi_data()
    pipa = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("seleksi", SelectKBest(f_classif)),
        ("model", SVC(random_state=RANDOM_STATE)),
    ])
    grid = {"seleksi__k": [10, 15, 20, 25, 30],
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.001, 0.01, 0.1]}
    gs = GridSearchCV(pipa, grid, cv=5, scoring="accuracy", n_jobs=-1).fit(Xtr, ytr)
    bp = gs.best_params_
    terbaik = model_svc_optimasi(bp["seleksi__k"], bp["model__C"], bp["model__gamma"]).fit(Xtr, ytr)
    sebelum = model_svc_optimasi(30, 1, "scale").fit(Xtr, ytr)     # 30 fitur, C=1 (default)
    hasil = {}
    for nama, m in [("Sebelum", sebelum), ("Sesudah", terbaik)]:
        pred, skor = m.predict(Xte), skor_model(m, Xte)
        hasil[nama] = hitung_metrik(yte, pred, skor)
    fitur_terpilih = list(Xtr.columns[terbaik.named_steps["seleksi"].get_support()])
    return {"gs": gs, "params": bp, "cv_terbaik": gs.best_score_, "model": terbaik,
            "hasil": hasil, "fitur_terpilih": fitur_terpilih,
            "tabel": pd.DataFrame(gs.cv_results_)}


@st.cache_data(show_spinner="Validasi silang 4 model...")
def hasil_cv():
    Xtr, _, ytr, _ = bagi_data()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    skor = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
            "F1": "f1", "ROC-AUC": "roc_auc"}
    baris = []
    for nama, m in model_empat().items():
        r = cross_validate(m, Xtr, ytr, cv=cv, scoring=skor)
        baris.append({"Model": nama,
                      **{k: r[f"test_{k}"].mean() for k in skor},
                      "F1 (std)": r["test_F1"].std()})
    return pd.DataFrame(baris).sort_values("F1", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Model untuk halaman prediksi
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Menyiapkan model prediksi...")
def model_prediksi():
    """Semua model dilatih pada data latih; dipakai oleh halaman prediksi tunggal & massal."""
    daftar = {"SVC RBF + optimasi (terbaik)": jalankan_optimasi()["model"]}
    dasar, _, _ = latih_model_dasar()
    daftar.update(dasar)
    return daftar


def pca_pipeline(n):
    return Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n, random_state=RANDOM_STATE))])


buat_pipa = _pipe   # alias publik


def vif_dari_korelasi(D):
    """VIF tiap kolom = elemen diagonal invers matriks korelasi."""
    c = np.corrcoef(D.T.values)
    return pd.Series(np.diag(np.linalg.inv(c)), index=D.columns)


def ringkas_kualitas_data():
    """Pemeriksaan kualitas data mentah (dipakai halaman Dataset & EDA)."""
    df = muat_data()
    X, y = siapkan_xy()
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    mask = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
    return {
        "baris": len(df), "kolom": df.shape[1],
        "missing": int(df.isna().sum().sum()),
        "duplikat": int(df.duplicated().sum()),
        "anomali": int((X[["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]] <= 0).any(axis=1).sum()),
        "nol_sel": int((X == 0).sum().sum()), "nol_persen": (X == 0).sum().sum() / X.size * 100,
        "nol_baris_concavity": int((X["concavity_mean"] == 0).sum()),
        "outlier_baris": int(mask.any(axis=1).sum()), "outlier_sel": int(mask.sum().sum()),
        "miring": int((X.skew().abs() > 1).sum()),
        "jinak": int((y == 0).sum()), "ganas": int((y == 1).sum()),
        "rasio": (y == 0).sum() / (y == 1).sum(),
    }
