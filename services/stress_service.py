import os
from pathlib import Path
from fastapi import HTTPException, status

from schemas.schemas import StressPredictRequest
from services.stress_model import StressModelService, StressModelUnavailableError, StressExplanationUnavailableError
from services.rag_pipeline import LocalHrvRagPipeline
from services.llm_explainer import StressExplanationLLM
from services.stress_analysis import StressAnalysisService

# Constants for paths
BASE_DIR = Path(__file__).resolve().parent.parent
_ML_ARTIFACT_PATH = BASE_DIR / "ml_models" / "all_artifacts.joblib"
_KNOWLEDGE_PATH = Path(os.getenv("HRV_KNOWLEDGE_PATH", str(BASE_DIR / "knowledge_base" / "hrv_medical_knowledge.json")))

# Instances
stress_model = StressModelService(_ML_ARTIFACT_PATH)
rag_pipeline = LocalHrvRagPipeline(
    knowledge_path=_KNOWLEDGE_PATH,
    collection_name=os.getenv("RAG_COLLECTION_NAME", "hrv_medical_knowledge"),
)
stress_analysis = StressAnalysisService(
    stress_model=stress_model,
    rag_pipeline=rag_pipeline,
    explanation_llm=StressExplanationLLM(),
)

class StressService:
    @staticmethod
    def predict_stress(body: StressPredictRequest):
        try:
            prediction = stress_model.predict(body)
        except StressModelUnavailableError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Stress prediction unavailable.",
            )
        return prediction

    @staticmethod
    def explain_stress(body: StressPredictRequest, top_n: int):
        try:
            prediction, explanation = stress_model.predict_with_explanation(
                body,
                top_n=top_n,
            )
        except StressModelUnavailableError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Stress prediction unavailable.",
            )
        except StressExplanationUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            )
        return prediction, explanation

    @staticmethod
    def analyze_stress(body: StressPredictRequest, top_features: int, top_k: int):
        try:
            result = stress_analysis.analyze(
                body,
                top_features=top_features,
                top_k=top_k,
            )
        except StressModelUnavailableError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Stress prediction unavailable.",
            )
        except StressExplanationUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            )
        return result
