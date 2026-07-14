import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# baca dataset
df = pd.read_csv('dataset_stunting_ml_1000.csv')
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# risk_score dibuang, kebukti data leakage (lihat analisis di notebook)
if 'risk_score' in df.columns:
    df = df.drop(columns=['risk_score'])

# pisahin fitur & kolom numerik/kategorikal
X = df.drop(columns=['status_stunting'])
cat_cols = ['jenis_kelamin', 'asi_eksklusif', 'sanitasi_layak', 'imunisasi_lengkap']
num_cols = [col for col in X.columns if col not in cat_cols]

# fit label encoder per kolom kategorikal
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(X[col])
    label_encoders[col] = le

# fit scaler buat kolom numerik
scaler = StandardScaler()
scaler.fit(X[num_cols])

# bundel semua preprocessor jadi satu file pkl
preprocessors = {
    'label_encoders': label_encoders,
    'scaler': scaler,
    'cat_cols': cat_cols,
    'num_cols': num_cols,
    'columns': list(X.columns)
}

joblib.dump(preprocessors, 'preprocessors.pkl')
print("Berhasil simpan preprocessors.pkl!")