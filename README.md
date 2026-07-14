Tentu, ini adalah file **`README.md`** versi final dan bersih yang sudah disesuaikan sepenuhnya dengan model perbaikan kamu (**No-Leakage / Tanpa `risk_score**`).

Format di bawah ini dibuat sangat terstruktur, profesional, namun tetap ringkas dan langsung ke intinya agar juri kompetisi langsung paham letak inovasi proyekmu:

```markdown
# 🧒 Stunting Prediction ML Project (No-Leakage Version)

Proyek Machine Learning untuk memprediksi status stunting pada balita menggunakan data fisik, gizi, klinis, dan sosial-ekonomi. Proyek ini merupakan hasil **improvement (perbaikan kritis)** dari model awal (versi mentor) dengan membuang fitur kebocoran data (*data leakage*) agar model dapat digunakan dengan aman, jujur, dan akurat pada aplikasi nyata (*production-ready*).

---

## 📁 Struktur Folder


```

ML/
├── stunting_ml_analysis_fixed_no_leakage.ipynb  # Notebook perbaikan, pelatihan ulang, dan evaluasi model (No-Leakage)
├── main.py                                      # API backend FastAPI untuk serving prediksi ke Laravel
├── stunting_model_fix_no_leakage.pkl            # File model ML terbaik versi perbaikan (tanpa risk_score)
├── dataset_stunting_ml_1000.csv                 # Dataset utama (1000 data balita)
├── generate_preprocessors.py                    # Script utilitas untuk generate preprocessors
├── requirements.txt                             # Daftar dependensi library Python
└── README.md                                    # Dokumentasi proyek ini

```

---

## 📄 Deskripsi File & Perubahan Penting

### 1. `stunting_ml_analysis_fixed_no_leakage.ipynb`
Jupyter Notebook utama yang berisi seluruh alur analisis data, pembuktian kebocoran data, dan pelatihan ulang model.

**Perbaikan Utama dibanding Versi Mentor:**
- **Penghapusan Fitur `risk_score` (Drop):** Berdasarkan uji korelasi formal, kolom `risk_score` terbukti merupakan *Data Leakage (Target Proxy)* yang merusak generalisasi model. Fitur ini dihapus agar model murni belajar dari data riil lapangan.
- **Uji Korelasi Formal:** Menambahkan visualisasi korelasi (Matriks Korelasi Pearson) untuk membuktikan secara ilmiah mengapa `risk_score` harus dibuang.
- **Perluasan Eksperimen Model:** Menguji lebih banyak algoritma klasifikasi (termasuk *Logistic Regression*, *Random Forest*, *Gradient Boosting*, dll.) dengan penanganan *imbalance data* yang lebih matang.
- **Evaluasi Berfokus pada Recall:** Model terbaik dipilih berdasarkan nilai *Recall* tertinggi untuk meminimalkan *False Negative* (balita stunting yang salah terprediksi sebagai normal).

---

### 2. `main.py`
Server API berbasis **FastAPI** yang bertugas menerima data dari aplikasi web Laravel, melakukan prapemrosesan, dan mengembalikan hasil prediksi stunting.

**Cara kerja:**
1. Saat server dijalankan, model baru `stunting_model_fix_no_leakage.pkl` di-load ke memori.
2. `LabelEncoder` dan `StandardScaler` di-*fit* ulang secara otomatis menggunakan `dataset_stunting_ml_1000.csv` (dengan kolom `risk_score` yang sudah diabaikan).
3. Saat menerima request `POST /predict`, data diubah melalui *pipeline* preprocessing (encoding → scaling → pengurutan kolom) tanpa membutuhkan input `risk_score`.

**Endpoint yang tersedia:**

| Method | Endpoint  | Deskripsi                               |
|--------|-----------|-----------------------------------------|
| `GET`  | `/`       | Cek status server (health check)        |
| `POST` | `/predict`| Mengirim data balita, mendapat prediksi |

**Contoh request body untuk `POST /predict` (Tanpa `risk_score`):**
```json
{
  "usia_bulan": 24,
  "jenis_kelamin": "L",
  "berat_lahir_kg": 3.2,
  "panjang_lahir_cm": 50.0,
  "asi_eksklusif": "Ya",
  "protein_harian": 45.0,
  "frekuensi_makan": 4,
  "tinggi_ibu_cm": 160.0,
  "riwayat_diare": 0,
  "pendapatan_keluarga": 6000000.0,
  "sanitasi_layak": "Ya",
  "imunisasi_lengkap": "Ya"
}

```

**Contoh response:**

```json
{
  "status": "success",
  "prediction_code": 0,
  "prediction_status": "Tidak Stunting",
  "probability_stunting_percent": 12.45
}

```

---

### 3. `stunting_model_fix_no_leakage.pkl`

File biner model final (Logistic Regression / Random Forest hasil revisi) yang disimpan menggunakan library `joblib`/`pickle`. Model ini **sangat stabil** karena memprediksi murni berdasarkan kondisi fisik, asupan gizi, dan riwayat kesehatan balita yang sebenarnya.

---

### 4. `dataset_stunting_ml_1000.csv`

Dataset yang digunakan untuk melatih model. Kolom `risk_score` diabaikan sepenuhnya selama proses pelatihan model revisi demi integritas data.

**Kolom Fitur yang Digunakan:**

| Kolom | Tipe | Deskripsi |
| --- | --- | --- |
| `id` | int | ID unik (diabaikan saat training) |
| `usia_bulan` | int | Usia balita dalam bulan |
| `jenis_kelamin` | string | Jenis kelamin: `L` (Laki-laki) / `P` (Perempuan) |
| `berat_lahir_kg` | float | Berat badan lahir (kilogram) |
| `panjang_lahir_cm` | float | Panjang badan lahir (sentimeter) |
| `asi_eksklusif` | string | Riwayat ASI eksklusif: `Ya` / `Tidak` |
| `protein_harian` | float | Asupan protein harian (gram) |
| `frekuensi_makan` | int | Frekuensi makan per hari |
| `tinggi_ibu_cm` | float | Tinggi badan ibu (sentimeter) |
| `riwayat_diare` | int | Frekuensi kejadian diare |
| `pendapatan_keluarga` | float | Pendapatan keluarga per bulan (Rupiah) |
| `sanitasi_layak` | string | Akses sanitasi layak: `Ya` / `Tidak` |
| `imunisasi_lengkap` | string | Status imunisasi lengkap: `Ya` / `Tidak` |
| `status_stunting` | int | **Target:** `1` = Stunting, `0` = Tidak Stunting |

---

## 🚀 Cara Instalasi dan Menjalankan API

### 1. Instalasi Dependensi

Arahkan terminal ke dalam folder `ML/` dan jalankan perintah:

```bash
pip install -r requirements.txt

```

### 2. Menjalankan Server

Jalankan perintah berikut untuk mengaktifkan API backend pada port `8001`:

```bash
uvicorn main:app --reload --port 8001

```

Server berjalan aktif di: **`http://127.0.0.1:8001`** Akses Swagger UI (Dokumentasi API interaktif) di: **`http://127.0.0.1:8001/docs`**

---

## 🔗 Integrasi dengan Laravel (Tanpa `risk_score`)

Pada Controller Laravel Anda, gunakan **HTTP Client** bawaan untuk menembak API FastAPI. Perhatikan bahwa Anda **tidak perlu mengirim parameter `risk_score**` karena model baru sudah mampu memprediksi dengan akurat tanpa kolom tersebut.

```php
use Illuminate\Support\Facades\Http;

$response = Http::timeout(10)->post('[http://127.0.0.1:8001/predict](http://127.0.0.1:8001/predict)', [
    'usia_bulan'          => 24,
    'jenis_kelamin'       => 'L',
    'berat_lahir_kg'      => 3.2,
    'panjang_lahir_cm'    => 50.0,
    'asi_eksklusif'       => 'Ya',
    'protein_harian'      => 45.0,
    'frekuensi_makan'     => 4,
    'tinggi_ibu_cm'       => 160.0,
    'riwayat_diare'       => 0,
    'pendapatan_keluarga' => 6000000.0,
    'sanitasi_layak'      => 'Ya',
    'imunisasi_lengkap'   => 'Ya',
]);

$result =$response->json();
// Hasil prediksi:
// $result['prediction_status'] => "Tidak Stunting"

```

---

## 🏗️ Arsitektur Sistem Baru (No-Leakage)

```
[ Form Input Kader di Laravel ]
               |
               | POST (Data fisik & gizi riil balita)
               ▼
[ Laravel Controller ]
               |
               | Http::post (Mengirim 12 fitur bersih ke port 8001)
               ▼
[ FastAPI Server (main.py) ]
               |
               | 1. Encoding & Scaling otomatis (Tanpa risk_score)
               | 2. Prediksi jujur via stunting_model_fix_no_leakage.pkl
               ▼
[ Response JSON (Akurat & Stabil) ] ──▶ [ Ditampilkan ke Layar Laravel ]

```

```

```