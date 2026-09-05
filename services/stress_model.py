import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from core.logger import get_logger
logger = get_logger(__name__)

DEMOGRAPHIC_FEATURE_FIELDS = ("age", "gender_male", "height_cm", "weight_kg")
HRV_FEATURE_FIELDS = (
    "sdnn",
    "median_rr",
    "cv_rr",
    "rmssd",
    "pnn50",
    "pnn20",
    "mean_hr",
    "std_hr",
    "min_hr",
    "max_hr",
    "hr_range",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "total_power",
    "lf_norm",
    "sd1",
    "sd2",
    "sd_ratio",
)
DEFAULT_FEATURE_COLUMNS = (*DEMOGRAPHIC_FEATURE_FIELDS, *HRV_FEATURE_FIELDS)

FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "gender_male": "Male gender indicator",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "sdnn": "SDNN",
    "median_rr": "Median RR interval",
    "cv_rr": "RR coefficient of variation",
    "rmssd": "RMSSD",
    "pnn50": "pNN50",
    "pnn20": "pNN20",
    "mean_hr": "Mean heart rate",
    "std_hr": "Heart-rate variability",
    "min_hr": "Minimum heart rate",
    "max_hr": "Maximum heart rate",
    "hr_range": "Heart-rate range",
    "lf_power": "LF power",
    "hf_power": "HF power",
    "lf_hf_ratio": "LF/HF ratio",
    "total_power": "Total spectral power",
    "lf_norm": "Normalized LF power",
    "sd1": "Poincare SD1",
    "sd2": "Poincare SD2",
    "sd_ratio": "SD2/SD1 ratio",
}


class StressModelUnavailableError(RuntimeError):
    pass


class StressExplanationUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class StressPrediction:
    stress_level_pct: float
    is_stressed: bool
    feature_values: dict[str, float]


@dataclass(frozen=True)
class ShapFeatureContribution:
    feature: str
    display_name: str
    value: float
    shap_value: float
    contribution_pct: float
    direction: str


@dataclass(frozen=True)
class ShapExplanation:
    model_output: str
    base_value: float
    base_value_pct: float
    prediction_value: float
    prediction_pct: float
    top_contributions: list[ShapFeatureContribution]


class StressModelService:
    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self._artifacts: Optional[dict[str, Any]] = None
        self._model: Any = None
        self._feature_columns = list(DEFAULT_FEATURE_COLUMNS)
        self._explainer: Any = None
        self._shap_lock = threading.Lock()
        self.load()

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def load(self) -> None:
        try:
            artifacts = joblib.load(self.artifact_path)
            self._artifacts = artifacts
            self._model = artifacts["model"]
            self._feature_columns = list(artifacts["feature_columns"])
            logger.info("Stress ML model loaded from %s", self.artifact_path)
        except Exception as exc:
            logger.warning(
                "Stress ML model not found at %s; prediction unavailable: %s",
                self.artifact_path,
                exc,
            )

    def build_features(self, request: Any, user: Any = None) -> dict[str, float]:
        artifacts = self._artifacts or {}
        demo_defaults = artifacts.get("demo_defaults", {})
        features = {key: float(value) for key, value in demo_defaults.items()}

        for field_name in HRV_FEATURE_FIELDS:
            features[field_name] = self._to_float(getattr(request, field_name), field_name)

        for field_name in DEMOGRAPHIC_FEATURE_FIELDS:
            body_value = getattr(request, field_name)
            if body_value is not None:
                features[field_name] = self._to_float(body_value, field_name)

        if user is not None:
            if (
                getattr(request, "age") is None
                and getattr(user, "age", None) is not None
            ):
                features["age"] = float(user.age)
            if (
                getattr(request, "gender_male") is None
                and getattr(user, "gender", None) is not None
            ):
                features["gender_male"] = 1.0 if user.gender == "male" else 0.0
            if (
                getattr(request, "height_cm") is None
                and getattr(user, "height_cm", None) is not None
            ):
                features["height_cm"] = float(user.height_cm)
            if (
                getattr(request, "weight_kg") is None
                and getattr(user, "weight_kg", None) is not None
            ):
                features["weight_kg"] = float(user.weight_kg)

        for field_name in self._feature_columns:
            features.setdefault(field_name, 0.0)

        return features

    def predict(self, request: Any, user: Any = None) -> StressPrediction:
        self._require_model()
        features = self.build_features(request, user)
        stress_probability = self._predict_stress_probability(features)
        stress_pct = round(stress_probability * 100, 1)
        return StressPrediction(
            stress_level_pct=stress_pct,
            is_stressed=stress_pct >= 50,
            feature_values=features,
        )

    def predict_with_explanation(
        self,
        request: Any,
        user: Any = None,
        top_n: int = 8,
    ) -> tuple[StressPrediction, ShapExplanation]:
        prediction = self.predict(request, user)
        explanation = self.explain(prediction.feature_values, top_n=top_n)
        return prediction, explanation

    def explain(self, features: dict[str, float], top_n: int = 8) -> ShapExplanation:
        self._require_model()
        feature_matrix = self._feature_matrix(features)
        prediction_value = self._predict_stress_probability(features)
        shap_values, expected_value = self._compute_shap_values(feature_matrix)
        positive_class_values = self._positive_class_shap_values(shap_values)
        base_value = self._positive_class_expected_value(expected_value)

        contributions = []
        for feature_name, shap_value in zip(self._feature_columns, positive_class_values):
            contribution = float(shap_value)
            contributions.append(
                ShapFeatureContribution(
                    feature=feature_name,
                    display_name=FEATURE_DISPLAY_NAMES.get(
                        feature_name,
                        feature_name.replace("_", " ").title(),
                    ),
                    value=round(float(features.get(feature_name, 0.0)), 6),
                    shap_value=round(contribution, 6),
                    contribution_pct=round(contribution * 100, 3),
                    direction=self._direction(contribution),
                )
            )

        contributions.sort(key=lambda item: abs(item.shap_value), reverse=True)
        top_count = max(1, min(top_n, len(contributions)))

        return ShapExplanation(
            model_output="stress_probability",
            base_value=round(base_value, 6),
            base_value_pct=round(base_value * 100, 3),
            prediction_value=round(prediction_value, 6),
            prediction_pct=round(prediction_value * 100, 1),
            top_contributions=contributions[:top_count],
        )

    def _require_model(self) -> None:
        if self._model is None:
            raise StressModelUnavailableError("Stress prediction unavailable.")

    def _feature_matrix(self, features: dict[str, float]) -> np.ndarray:
        return np.array(
            [[float(features.get(column, 0.0)) for column in self._feature_columns]],
            dtype=float,
        )

    def _predict_stress_probability(self, features: dict[str, float]) -> float:
        probabilities = self._model.predict_proba(self._feature_matrix(features))[0]
        return float(probabilities[1])

    def _compute_shap_values(self, feature_matrix: np.ndarray) -> tuple[Any, Any]:
        try:
            import shap
        except Exception as exc:
            raise StressExplanationUnavailableError(
                "SHAP is not installed. Install the 'shap' package to enable explanations."
            ) from exc

        with self._shap_lock:
            if self._explainer is None:
                clf = getattr(self._model, "named_steps", {}).get("clf", self._model)
                try:
                    self._explainer = shap.TreeExplainer(clf)
                except Exception as exc:
                    logger.warning(
                        "SHAP explainer unavailable for %s: %s", type(clf).__name__, exc
                    )
                    raise StressExplanationUnavailableError(
                        "Explanations require a tree-based model; the loaded model is "
                        f"{type(clf).__name__}."
                    ) from exc

            scaler = getattr(self._model, "named_steps", {}).get("scaler")
            scaled_features = scaler.transform(feature_matrix) if scaler else feature_matrix

            try:
                shap_values = self._explainer.shap_values(scaled_features)
            except Exception as exc:
                logger.warning("SHAP computation failed: %s", exc)
                raise StressExplanationUnavailableError(
                    "Failed to compute SHAP explanation."
                ) from exc
            return shap_values, self._explainer.expected_value

    def _positive_class_shap_values(self, shap_values: Any) -> np.ndarray:
        feature_count = len(self._feature_columns)

        if isinstance(shap_values, list):
            selected = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            selected_array = np.asarray(selected, dtype=float)
            return selected_array[0] if selected_array.ndim == 2 else selected_array

        values = np.asarray(shap_values, dtype=float)
        if values.ndim == 1:
            return values
        if values.ndim == 2:
            return values[0]
        if values.ndim == 3:
            if values.shape[0] == 1 and values.shape[1] == feature_count:
                return values[0, :, 1]
            if values.shape[0] >= 2 and values.shape[2] == feature_count:
                return values[1, 0, :]

        raise StressExplanationUnavailableError("Unexpected SHAP values shape.")

    def _positive_class_expected_value(self, expected_value: Any) -> float:
        expected_values = np.asarray(expected_value, dtype=float)
        if expected_values.ndim == 0:
            return float(expected_values)
        return float(
            expected_values[1]
            if expected_values.shape[0] > 1
            else expected_values[0]
        )

    @staticmethod
    def _to_float(value: Any, field_name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric") from exc

    @staticmethod
    def _direction(value: float) -> str:
        if value > 1e-9:
            return "increases_stress"
        if value < -1e-9:
            return "decreases_stress"
        return "neutral"
