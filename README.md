# 🏥 Stunting Prediction ML Project

Proyek Machine Learning untuk memprediksi status stunting pada balita menggunakan data klinis dan sosial-ekonomi. Proyek ini terdiri dari notebook analisis/pelatihan model, API backend berbasis FastAPI, dan dataset pendukung.

---

## 📁 Struktur Folder

```
ML/
├── stunting_ml_analysis.ipynb   # Notebook eksplorasi, pelatihan, dan evaluasi model
├── main.py                      # API backend FastAPI untuk serving prediksi
├── best_model.pkl               # File model ML terbaik (hasil training)
├── dataset_stunting_ml_1000.csv # Dataset utama (1000 data balita)
├── generate_preprocessors.py    # Script utilitas untuk generate preprocessors
├── requirements.txt             # Daftar dependensi library Python
└── README.md                    # Dokumentasi proyek ini
```

---

## 📄 Deskripsi File

### 1. `stunting_ml_analysis.ipynb`
Jupyter Notebook utama yang berisi seluruh alur analisis data dan pelatihan model dari awal hingga akhir.

**Isi notebook ini meliputi:**
- **EDA (Exploratory Data Analysis):** Eksplorasi distribusi data, deteksi missing value, dan visualisasi statistik deskriptif.
- **Preprocessing:** Label Encoding untuk kolom kategorikal (`jenis_kelamin`, `asi_eksklusif`, `sanitasi_layak`, `imunisasi_lengkap`) dan StandardScaler untuk kolom numerik.
- **Pelatihan 3 Model:** Logistic Regression, Random Forest, dan Gradient Boosting.
- **Evaluasi:** Perbandingan performa model menggunakan Accuracy, Precision, Recall, F1-Score, dan ROC-AUC. Model terbaik dipilih berdasarkan F1-Score tertinggi.
- **Feature Importance:** Analisis fitur paling berpengaruh menggunakan feature importance bawaan dan Permutation Importance.
- **Contoh Prediksi:** Prediksi pada 12 data balita baru dengan berbagai variasi kondisi.

---

### 2. `main.py`
Server API berbasis **FastAPI** yang bertugas menerima data dari aplikasi eksternal (misal Laravel), memprosesnya, dan mengembalikan hasil prediksi stunting.

**Cara kerja:**
1. Saat server dinyalakan, model `best_model.pkl` di-load ke memori.
2. `LabelEncoder` dan `StandardScaler` di-*fit* ulang secara otomatis menggunakan `dataset_stunting_ml_1000.csv` (ini menggantikan encoder/scaler yang tidak ikut tersimpan di dalam `.pkl`).
3. Saat menerima request `POST /predict`, data input diproses (encoding → scaling → pengurutan kolom) sebelum diprediksi oleh model.

**Endpoint yang tersedia:**

| Method | Endpoint  | Deskripsi                              |
|--------|-----------|----------------------------------------|
| `GET`  | `/`       | Cek status server (health check)       |
| `POST` | `/predict`| Mengirim data balita, mendapat prediksi|

**Contoh request body untuk `POST /predict`:**
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
  "imunisasi_lengkap": "Ya",
  "risk_score": 15.0
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

> **Catatan nilai untuk field string:**
> - `jenis_kelamin`: `"L"` atau `"P"`
> - `asi_eksklusif`, `sanitasi_layak`, `imunisasi_lengkap`: `"Ya"` atau `"Tidak"`

---

### 3. `best_model.pkl`
File biner yang berisi model ML terbaik hasil proses pelatihan di `stunting_ml_analysis.ipynb`, disimpan menggunakan library `joblib`.

> ⚠️ **Penting:** File ini **hanya** berisi objek model (classifier) tanpa encoder/scaler. Preprocessing (Label Encoding & Standard Scaling) dilakukan secara terpisah di `main.py` menggunakan dataset asli.

---

### 4. `dataset_stunting_ml_1000.csv`
Dataset utama yang berisi **1000 data rekam medis dan sosial-ekonomi balita**.

**Kolom yang tersedia:**

| Kolom                | Tipe    | Deskripsi                                        |
|----------------------|---------|--------------------------------------------------|
| `id`                 | int     | ID unik tiap data (tidak dipakai saat training)  |
| `usia_bulan`         | int     | Usia balita dalam bulan                          |
| `jenis_kelamin`      | string  | Jenis kelamin: `L` (Laki-laki) / `P` (Perempuan)|
| `berat_lahir_kg`     | float   | Berat badan lahir dalam kilogram                 |
| `panjang_lahir_cm`   | float   | Panjang badan lahir dalam sentimeter             |
| `asi_eksklusif`      | string  | Riwayat ASI eksklusif: `Ya` / `Tidak`            |
| `protein_harian`     | float   | Asupan protein harian (gram)                     |
| `frekuensi_makan`    | int     | Frekuensi makan per hari                         |
| `tinggi_ibu_cm`      | float   | Tinggi badan ibu dalam sentimeter                |
| `riwayat_diare`      | int     | Frekuensi/riwayat diare                          |
| `pendapatan_keluarga`| float   | Pendapatan keluarga per bulan (Rupiah)           |
| `sanitasi_layak`     | string  | Akses sanitasi layak: `Ya` / `Tidak`             |
| `imunisasi_lengkap`  | string  | Status imunisasi lengkap: `Ya` / `Tidak`         |
| `risk_score`         | float   | Skor risiko terkomputasi                         |
| `status_stunting`    | int     | **Target:** `1` = Stunting, `0` = Tidak Stunting |

---

### 5. `generate_preprocessors.py`
Script utilitas Python untuk me-*fit* dan menyimpan preprocessors (`LabelEncoder` + `StandardScaler`) ke file `preprocessors.pkl`.

> 💡 Script ini berguna jika kamu ingin memisahkan file preprocessors agar tidak perlu membaca ulang CSV setiap kali server dijalankan. Saat ini, proses fitting dilakukan langsung di dalam `main.py` pada saat startup.

---

## 🚀 Cara Instalasi dan Menjalankan API

### 1. Prasyarat
Pastikan Python sudah terinstall di sistem Anda. Direkomendasikan untuk menggunakan *Virtual Environment* (misal `.venv`).

### 2. Instalasi Dependensi
Buka terminal, arahkan ke folder `ML/`, dan jalankan perintah berikut untuk menginstall seluruh library yang dibutuhkan berdasarkan file `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Menjalankan Server
Setelah proses instalasi selesai, Anda dapat menjalankan API dengan perintah berikut (di dalam folder `ML/`):

```bash
uvicorn main:app --reload --port 8001
```

> **Catatan Port:** API dijalankan pada port `8001` untuk menghindari bentrokan (*conflict*) dengan layanan lain yang mungkin menggunakan port default `8000`.

Server akan berjalan di: **`http://127.0.0.1:8001`**

Akses dokumentasi API interaktif (Swagger UI) di: **`http://127.0.0.1:8001/docs`**

---

## 🔗 Integrasi dengan Laravel

Di sisi Laravel, gunakan **HTTP Client** bawaan (`Illuminate\Support\Facades\Http`) untuk mengirim data ke API dan menerima hasilnya. Pastikan URL endpoint mengarah ke port **8001**.

```php
use Illuminate\Support\Facades\Http;

$response = Http::timeout(10)->post('http://127.0.0.1:8001/predict', [
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
    'risk_score'          => 15.0,
]);

$result = $response->json();
// $result['prediction_status'] => "Tidak Stunting"
// $result['probability_stunting_percent'] => 12.45
```

---

## 🏗️ Arsitektur Sistem

```
[ Browser / Form Laravel ]
         |
         | POST (form input data balita)
         ▼
[ Laravel Controller ]
         |
         | Http::post ke port 8001
         ▼
[ FastAPI Server (main.py) ]
         |
         | 1. Label Encoding (kategorikal)
         | 2. Standard Scaling (numerik)
         | 3. Urutkan kolom
         | 4. model.predict()
         ▼
[ Response JSON ke Laravel ]
         |
         ▼
[ Tampilkan Hasil ke User ]
```
