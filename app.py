"""
Aplikasi Streamlit: Klasifikasi Anomali dan Serangan Jaringan (UNSW-NB15) dengan Random Forest
Proyek 01, Machine Learning for Network Data, D4 TRKJ Universitas Bung Hatta

Jalankan lokal:  streamlit run app.py

DAFTAR MODIFIKASI DARI TEMPLATE:
1. Sidebar identitas mahasiswa (ganti Alip Farhandana Lubis dan 2410017514004 di bawah)
2. Judul halaman, ikon, dan deskripsi diubah menjadi "NetGuard"
3. Fitur baru: tabel 10 flow paling mencurigakan di tab CSV
4. Threshold default diubah dari 0.50 menjadi 0.40
5. Tab baru "Kategori serangan" (tab keempat)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
SAMPLE_CSV = BASE / "data" / "sample_flows.csv"

# [MODIFIKASI 1] Isi identitas kamu di sini
NAMA_MAHASISWA = "Nama Kamu"
NIM_MAHASISWA = "1234567890"

# [MODIFIKASI 2] Judul tab browser dan ikon diubah
st.set_page_config(page_title="NetGuard - Deteksi Serangan UNSW-NB15", page_icon="🔐", layout="wide")


# ---------- Muat model (sekali saja, lalu di-cache) ----------
@st.cache_resource
def load_models():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    rf_bin = joblib.load(MODEL_DIR / "rf_binary.joblib")
    rf_cat = joblib.load(MODEL_DIR / "rf_multiclass.joblib")
    return rf_bin, rf_cat, meta


rf_bin, rf_cat, meta = load_models()
FEATURES = meta["features"]


def predict(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Lengkapi kolom yang hilang dengan nilai default, lalu prediksi dengan kedua model."""
    X = df.copy()
    for col in FEATURES:
        if col not in X.columns:
            X[col] = meta["defaults"][col]
    X = X[FEATURES]
    for col in meta["cat_cols"]:
        X[col] = X[col].astype(str)
    prob = rf_bin.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "prob_serangan": prob.round(3),
        "prediksi": ["SERANGAN" if p >= threshold else "normal" for p in prob],
        "kategori_prediksi": rf_cat.predict(X),
    })
    return out


# ---------- Sidebar ----------
with st.sidebar:
    # [MODIFIKASI 1] Identitas mahasiswa
    st.header("👤 Identitas")
    st.write(f"Nama: **{NAMA_MAHASISWA}**")
    st.write(f"NIM: **{NIM_MAHASISWA}**")
    st.write("D4 TRKJ, Universitas Bung Hatta")
    st.divider()

    st.header("⚙️ Pengaturan")
    # [MODIFIKASI 4] Nilai default threshold 0.50 -> 0.40
    threshold = st.slider("Threshold probabilitas serangan", 0.05, 0.95, 0.40, 0.05,
                          help="Flow dianggap serangan bila probabilitasnya ≥ threshold. "
                               "Turunkan untuk menangkap lebih banyak serangan (recall naik), "
                               "dengan risiko alarm palsu bertambah.")
    st.divider()
    st.subheader("Performa model (data test)")
    mb, mc = meta["metrics_binary"], meta["metrics_multiclass"]
    c1, c2 = st.columns(2)
    c1.metric("Akurasi biner", f"{mb['accuracy']:.1%}")
    c2.metric("ROC-AUC", f"{mb['roc_auc']:.3f}")
    c1.metric("Recall serangan", f"{mb['recall']:.1%}")
    c2.metric("Akurasi kategori", f"{mc['accuracy']:.1%}")
    st.caption(f"Dataset: {meta['dataset']}")

# ---------- Halaman utama ----------
# [MODIFIKASI 2] Judul dan deskripsi diubah
st.title("🔐 NetGuard: Deteksi Anomali dan Serangan Jaringan")
st.write("**NetGuard** memakai model **Random Forest** yang dilatih pada dataset **UNSW-NB15** untuk "
         "memprediksi apakah sebuah flow jaringan normal atau serangan, sekaligus menebak kategori "
         "serangannya.")

# [MODIFIKASI 5] Tab baru "Kategori Serangan" ditambahkan sebagai tab keempat
tab_csv, tab_manual, tab_info, tab_kat = st.tabs(
    ["📄 Prediksi dari CSV", "✍️ Input manual", "ℹ️ Tentang model", "📚 Kategori serangan"])

# --- Tab 1: CSV ---
with tab_csv:
    st.write("Unggah file CSV berisi flow dengan kolom seperti dataset UNSW-NB15, "
             "atau pakai 300 flow contoh dari data test.")
    uploaded = st.file_uploader("File CSV", type="csv")
    use_sample = st.toggle("Gunakan data contoh", value=uploaded is None)

    df = None
    if uploaded is not None and not use_sample:
        df = pd.read_csv(uploaded)
    elif use_sample:
        df = pd.read_csv(SAMPLE_CSV)

    if df is not None:
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.warning(f"{len(missing)} kolom tidak ada dan diisi nilai default (median data latih): "
                       f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
        res = predict(df, threshold)
        shown = pd.concat([res, df.drop(columns=[c for c in res.columns if c in df.columns])], axis=1)

        n_att = int((res["prediksi"] == "SERANGAN").sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Jumlah flow", len(res))
        k2.metric("Diprediksi serangan", n_att)
        k3.metric("Persentase serangan", f"{n_att / len(res):.1%}")

        if "label" in df.columns:
            acc = ((res["prediksi"] == "SERANGAN").astype(int) == df["label"]).mean()
            st.info(f"File memuat kolom `label`, jadi prediksi bisa dicek: akurasi biner = **{acc:.1%}**")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Kategori yang diprediksi")
            kat = res["kategori_prediksi"].value_counts().rename_axis("kategori").reset_index(name="jumlah")
            st.bar_chart(kat, x="kategori", y="jumlah", horizontal=True, sort="-jumlah")
        with right:
            st.subheader("Sebaran probabilitas serangan")
            st.bar_chart(pd.cut(res["prob_serangan"], bins=[0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
                         .astype(str).value_counts().sort_index())

        # [MODIFIKASI 3] Fitur baru: 10 flow dengan probabilitas serangan tertinggi
        st.subheader("🔥 10 flow paling mencurigakan")
        st.caption("Diurutkan dari probabilitas serangan tertinggi, berguna untuk prioritas analisis.")
        st.dataframe(shown.sort_values("prob_serangan", ascending=False).head(10),
                     width="stretch")

        st.subheader("Hasil per flow")
        st.dataframe(shown, width="stretch", height=360)
        st.download_button("⬇️ Unduh hasil (CSV)", shown.to_csv(index=False).encode(),
                           "hasil_prediksi.csv", "text/csv")

# --- Tab 2: Input manual ---
with tab_manual:
    st.write("Pilih flow contoh sebagai titik awal, lalu ubah fitur-fitur terpenting dan tekan **Prediksi**. "
             "Fitur yang tidak ditampilkan memakai nilai dari flow contoh tersebut.")
    samples = pd.read_csv(SAMPLE_CSV)
    pilihan = st.selectbox("Mulai dari contoh flow kategori", sorted(samples["attack_cat"].unique()),
                           index=sorted(samples["attack_cat"].unique()).index("Normal"))
    base = samples[samples["attack_cat"] == pilihan].iloc[0]
    with st.form("manual"):
        cols = st.columns(3)
        row = {c: base[c] for c in FEATURES}
        for i, c in enumerate(meta["cat_cols"]):
            opts = sorted(set(meta["cat_options"][c]) | {str(base[c])})
            row[c] = cols[i].selectbox(c, opts, index=opts.index(str(base[c])), key=f"{pilihan}-{c}")
        cols = st.columns(4)
        for i, c in enumerate(meta["top_numeric"]):
            lo, hi = meta["num_ranges"][c]
            row[c] = cols[i % 4].number_input(c, min_value=float(lo), value=float(base[c]), key=f"{pilihan}-{c}",
                                              help=f"Rentang umum data latih: {lo:g} sampai {hi:g}")
        submitted = st.form_submit_button("🔍 Prediksi", type="primary")
    if submitted:
        r = predict(pd.DataFrame([row]), threshold).iloc[0]
        if r["prediksi"] == "SERANGAN":
            st.error(f"**SERANGAN** terdeteksi (probabilitas {r['prob_serangan']:.1%}). "
                     f"Kategori paling mungkin: **{r['kategori_prediksi']}**")
        else:
            st.success(f"**Normal** (probabilitas serangan {r['prob_serangan']:.1%})")

# --- Tab 3: Info model ---
with tab_info:
    st.subheader("Fitur terpenting (model biner)")
    imp = pd.Series(meta["importance_top15"]).rename_axis("fitur").reset_index(name="importance")
    st.bar_chart(imp, x="fitur", y="importance", horizontal=True, sort="-importance")
    st.subheader("Metrik pada data test")
    st.table(pd.DataFrame({
        "Model biner": {k: round(v, 4) for k, v in meta["metrics_binary"].items()},
    }))
    st.table(pd.DataFrame({
        "Model multikelas": {k: round(v, 4) for k, v in meta["metrics_multiclass"].items()},
    }))
    st.caption(f"Dilatih dengan Python {meta['versions']['python']}, "
               f"scikit-learn {meta['versions']['scikit-learn']}, numpy {meta['versions']['numpy']}.")

# --- Tab 4: Kategori serangan ---
# [MODIFIKASI 5] Tab baru: penjelasan kategori serangan UNSW-NB15
with tab_kat:
    st.subheader("Kategori serangan pada dataset UNSW-NB15")
    st.write("Model multikelas menebak salah satu kategori berikut. "
             "Kolom jumlah menunjukkan banyaknya flow tiap kategori pada data contoh.")
    DESKRIPSI = {
        "Normal": "Lalu lintas jaringan biasa, bukan serangan.",
        "Fuzzers": "Mengirim data acak atau tidak valid untuk membuat sistem error atau crash.",
        "Analysis": "Pemindaian dan penyusupan analisis, seperti port scan, spam, dan skrip web berbahaya.",
        "Backdoor": "Melewati autentikasi normal agar penyerang punya akses tersembunyi ke sistem.",
        "DoS": "Denial of Service: membanjiri layanan agar tidak bisa diakses pengguna sah.",
        "Exploits": "Memanfaatkan celah atau bug pada sistem atau aplikasi.",
        "Generic": "Serangan terhadap kriptografi yang bekerja pada semua jenis block cipher.",
        "Reconnaissance": "Pengintaian: mengumpulkan informasi tentang target sebelum menyerang.",
        "Shellcode": "Menyuntikkan kode kecil untuk mengambil kendali mesin target.",
        "Worms": "Program yang menyebar sendiri ke komputer lain di jaringan.",
    }
    jumlah = pd.read_csv(SAMPLE_CSV)["attack_cat"].value_counts()
    tabel = pd.DataFrame({
        "Kategori": list(DESKRIPSI.keys()),
        "Deskripsi": list(DESKRIPSI.values()),
        "Jumlah di data contoh": [int(jumlah.get(k, 0)) for k in DESKRIPSI],
    })
    st.dataframe(tabel, hide_index=True, width="stretch")

    pilih = st.selectbox("Lihat detail kategori", list(DESKRIPSI.keys()), key="kat_detail")
    st.info(f"**{pilih}**: {DESKRIPSI[pilih]}")
