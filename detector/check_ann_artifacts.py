from pathlib import Path
import joblib

ARTIFACTS = Path(__file__).resolve().parent / "ml_artifacts"
scaler = joblib.load(ARTIFACTS / "scaler_ann.pkl")
print("Nombre de features du scaler :", scaler.n_features_in_)
print("Features :", list(scaler.feature_names_in_))
print("Seuil :", joblib.load(ARTIFACTS / "ann_threshold.pkl"))
