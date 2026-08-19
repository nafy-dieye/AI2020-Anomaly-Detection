import math
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent / "ml_artifacts"

RAW_INPUT_FIELDS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

FIELD_LABELS = {
    "air_temperature": ("Température de l'air", "K"),
    "process_temperature": ("Température du process", "K"),
    "rotational_speed": ("Vitesse de rotation", "rpm"),
    "torque": ("Couple", "Nm"),
    "tool_wear": ("Usure de l'outil", "min"),
}

# IMPORTANT : le fichier ann_model.keras fourni avec ce projet
# attend 11 variables et le scaler_ann.pkl fourni contient également
# la variable Type en première position.
# Encodage utilisé par cet artefact : H=0, L=1, M=2.
AI4I_TYPE_ENCODING = {"H": 0.0, "L": 1.0, "M": 2.0}


def engineer_features(raw: dict) -> dict:
    air_temp = raw["air_temperature"]
    process_temp = raw["process_temperature"]
    rpm = raw["rotational_speed"]
    torque = raw["torque"]
    tool_wear = raw["tool_wear"]

    angular_velocity = rpm * 2 * math.pi / 60

    return {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "temperature_difference": process_temp - air_temp,
        "mechanical_power": torque * angular_velocity,
        "tool_stress": torque * tool_wear,
        "temperature_ratio": process_temp / air_temp if air_temp else 0.0,
        "torque_speed_ratio": torque / rpm if rpm else 0.0,
    }


class AnomalyModel:
    # Auto-encodeur Keras + scaler + seuil.
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(
            ARTIFACTS_DIR / "autoencoder.keras"
        )
        self.scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")

        threshold_data = joblib.load(
            ARTIFACTS_DIR / "threshold.pkl"
        )
        self.threshold = float(threshold_data["threshold"])
        self.threshold_percentile = threshold_data.get("percentile")
        self.feature_names = joblib.load(
            ARTIFACTS_DIR / "feature_names.pkl"
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def predict(self, raw: dict) -> dict:
        features = engineer_features(raw)

        X = pd.DataFrame(
            [features],
            columns=self.feature_names,
        )

        X_scaled = self.scaler.transform(X)
        X_reconstructed = self.model.predict(
            X_scaled,
            verbose=0,
        )

        score = float(
            np.mean((X_scaled - X_reconstructed) ** 2)
        )
        is_anomaly = score > self.threshold

        return {
            "model_used": "autoencoder",
            "model_display_name": "Auto-encodeur",
            "score_label": "Erreur de reconstruction",
            "score": score,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "is_anomaly": is_anomaly,
            "severity_ratio": (
                score / self.threshold
                if self.threshold
                else None
            ),
            "engineered_features": features,
        }


class ANNModel:
    # ANN supervisé Keras + scaler + seuil.
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(
            ARTIFACTS_DIR / "ann_model.keras"
        )
        self.scaler = joblib.load(
            ARTIFACTS_DIR / "scaler_ann.pkl"
        )

        threshold_data = joblib.load(
            ARTIFACTS_DIR / "ann_threshold.pkl"
        )
        self.threshold = float(threshold_data["threshold"])
        self.threshold_percentile = threshold_data.get("percentile")

        self.feature_names = list(
            self.scaler.feature_names_in_
        )

        # Vérification de cohérence des artefacts au démarrage.
        # Le modèle fourni attend actuellement 11 entrées, avec Type.
        expected = [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "temperature_difference",
            "mechanical_power",
            "tool_stress",
            "temperature_ratio",
            "torque_speed_ratio",
        ]
        if self.feature_names != expected:
            raise ValueError(
                "Incohérence entre scaler_ann.pkl et l'intégration Django : "
                f"features reçues={self.feature_names}, attendu={expected}."
            )

        if getattr(self.model, "input_shape", None)[-1] != len(self.feature_names):
            raise ValueError(
                "ann_model.keras et scaler_ann.pkl n'ont pas le même nombre "
                "de variables d'entrée."
            )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def predict(self, raw: dict, machine_type: str = "M") -> dict:
        machine_type = str(machine_type).upper()

        if machine_type not in AI4I_TYPE_ENCODING:
            raise ValueError(
                "Type de machine ANN invalide. Utilisez H, L ou M."
            )

        features = engineer_features(raw)

        # L'ANN attend 11 features, dont Type.
        features["Type"] = AI4I_TYPE_ENCODING[machine_type]

        X = pd.DataFrame(
            [features],
            columns=self.feature_names,
        )

        X_scaled = self.scaler.transform(X)

        score = float(
            self.model.predict(
                X_scaled,
                verbose=0,
            ).reshape(-1)[0]
        )

        is_anomaly = score >= self.threshold

        return {
            "model_used": "ann",
            "model_display_name": "ANN",
            "score_label": "Score ANN",
            "score": score,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "is_anomaly": is_anomaly,
            "severity_ratio": (
                score / self.threshold
                if self.threshold
                else None
            ),
            "machine_type": machine_type,
            "engineered_features": features,
        }
