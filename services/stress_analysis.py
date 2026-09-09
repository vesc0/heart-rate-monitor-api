from dataclasses import dataclass
from typing import Any

from services.llm_explainer import StressExplanationLLM
from services.rag_pipeline import LocalHrvRagPipeline, RetrievedContext
from services.stress_model import ShapFeatureContribution, StressModelService


@dataclass(frozen=True)
class StressAnalysisResult:
    stress_level_pct: float
    is_stressed: bool
    important_features: list[ShapFeatureContribution]
    retrieved_context: list[RetrievedContext]
    explanation: str


class StressAnalysisService:
    def __init__(
        self,
        stress_model: StressModelService,
        rag_pipeline: LocalHrvRagPipeline,
        explanation_llm: StressExplanationLLM,
    ):
        self._stress_model = stress_model
        self._rag_pipeline = rag_pipeline
        self._explanation_llm = explanation_llm

    def analyze(
        self,
        request: Any,
        top_features: int = 6,
        top_k: int = 4,
    ) -> StressAnalysisResult:
        prediction, shap_explanation = self._stress_model.predict_with_explanation(
            request,
            top_n=top_features,
        )
        important_features = shap_explanation.top_contributions
        retrieval_query = self._build_retrieval_query(
            prediction.feature_values,
            prediction.stress_level_pct,
            prediction.is_stressed,
            important_features,
        )
        retrieved_context = self._rag_pipeline.retrieve(retrieval_query, top_k=top_k)
        explanation = self._explanation_llm.generate(
            feature_values=prediction.feature_values,
            stress_level_pct=prediction.stress_level_pct,
            is_stressed=prediction.is_stressed,
            important_features=important_features,
            retrieved_context=retrieved_context,
        )

        return StressAnalysisResult(
            stress_level_pct=prediction.stress_level_pct,
            is_stressed=prediction.is_stressed,
            important_features=important_features,
            retrieved_context=retrieved_context,
            explanation=explanation,
        )

    @staticmethod
    def _build_retrieval_query(
        feature_values: dict[str, float],
        stress_level_pct: float,
        is_stressed: bool,
        important_features: list[ShapFeatureContribution],
    ) -> str:
        important_terms = " ".join(
            f"{feature.display_name} {feature.feature} {feature.direction}"
            for feature in important_features
        )
        key_values = (
            f"SDNN {feature_values.get('sdnn', 0.0):.3f} "
            f"RMSSD {feature_values.get('rmssd', 0.0):.3f} "
            f"pNN50 {feature_values.get('pnn50', 0.0):.3f} "
            f"mean heart rate {feature_values.get('mean_hr', 0.0):.3f} "
            f"LF/HF ratio {feature_values.get('lf_hf_ratio', 0.0):.3f} "
            f"SD1 {feature_values.get('sd1', 0.0):.3f} "
            f"SD2 {feature_values.get('sd2', 0.0):.3f}"
        )
        status = "stressed" if is_stressed else "not stressed"
        return (
            "HRV stress explanation autonomic nervous system sympathetic "
            "parasympathetic recovery PPG measurement "
            f"model stress {stress_level_pct:.1f}% {status}. "
            f"Important model features: {important_terms}. "
            f"Observed feature values: {key_values}."
        )
