import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import uvicorn

app = FastAPI(
    title="Stunting Prediction API",
    description="API untuk memprediksi status stunting menggunakan model machine learning.",
    version="1.0.0"
)

# Mendapatkan direktori absolut dari folder tempat script ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tentukan nama kolom kategorikal secara global agar bisa diakses di route /predict
cat_cols = ['jenis_kelamin', 'asi_eksklusif', 'sanitasi_layak', 'imunisasi_lengkap']

# load model pkl-nya pas aplikasi start menggunakan absolute path
try:
    model_path = os.path.join(BASE_DIR, "stunting_model_fix_no_leakage.pkl")
    best_model = joblib.load(model_path)
    print("✅ Berhasil load best_model.pkl")
except Exception as e:
    best_model = None
    print(f"❌ Gagal load model: {e}")

# fit ulang label encoder & scaler dari dataset training
print("Fitting preprocessors dari dataset...")
try:
    csv_path = os.path.join(BASE_DIR, "dataset_stunting_ml_1000.csv")
    df = pd.read_csv(csv_path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "risk_score" in df.columns:
        df = df.drop(columns=["risk_score"])

    X_train = df.drop(columns=["status_stunting"])

    num_cols = [col for col in X_train.columns if col not in cat_cols]

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X_train[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])

    expected_columns = list(X_train.columns)
    print("✅ Fitting preprocessors sukses!")

except Exception as e:
    print(f"❌ Gagal fitting preprocessors: {e}")
    label_encoders = None
    scaler = None
    expected_columns = None
    num_cols = []


# schema input dari Laravel
class PredictionInput(BaseModel):
    usia_bulan: int
    jenis_kelamin: str
    berat_lahir_kg: float
    panjang_lahir_cm: float
    asi_eksklusif: str
    protein_harian: float
    frekuensi_makan: int
    tinggi_ibu_cm: float
    riwayat_diare: int
    pendapatan_keluarga: float
    sanitasi_layak: str
    imunisasi_lengkap: str


@app.get("/")
def read_root():
    return {"message": "API Prediksi Stunting Aktif. Gunakan endpoint POST /predict untuk prediksi."}


@app.post("/predict")
def predict(data: PredictionInput):
    # Validasi apakah model dan preprocessor sudah siap digunakan
    if best_model is None or scaler is None or label_encoders is None or expected_columns is None:
        raise HTTPException(status_code=500, detail="Model atau preprocessor belum siap. Pastikan file .pkl dan .csv ada di folder yang benar.")

    try:
        # Mengubah data input ke DataFrame (gunakan dict() atau model_dump() tergantung versi pydantic)
        input_dict = data.dict() if hasattr(data, 'dict') else data.model_dump()
        new_data = pd.DataFrame([input_dict])
        new_data_processed = new_data.copy()

        # encode fitur kategorikal
        for col in cat_cols:
            if col in new_data_processed.columns:
                try:
                    new_data_processed[col] = label_encoders[col].transform(new_data_processed[col])
                except ValueError:
                    # kalau ada kategori aneh yang gak dikenal encoder, fallback ke 0
                    new_data_processed[col] = 0

        # scaling fitur numerik
        new_data_processed[num_cols] = scaler.transform(new_data_processed[num_cols])

        # urutan kolom disesuaikan dengan saat training
        new_data_processed = new_data_processed[expected_columns]

        # Lakukan prediksi
        pred_label = best_model.predict(new_data_processed)

        try:
            pred_proba = best_model.predict_proba(new_data_processed)[:, 1]
            prob_percent = round(float(pred_proba[0]) * 100, 2)
        except Exception:
            prob_percent = None

        result_label = int(pred_label[0])
        status = "Stunting" if result_label == 1 else "Tidak Stunting"

        return {
            "status": "success",
            "prediction_code": result_label,
            "prediction_status": status,
            "probability_stunting_percent": prob_percent
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Terjadi kesalahan saat memproses prediksi: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)