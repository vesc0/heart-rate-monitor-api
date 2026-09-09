from dataclasses import asdict
from fastapi import APIRouter, Depends, Query

from api.dependencies import get_current_user_id
from schemas.schemas import StressPredictRequest, StressPredictResponse, StressExplainResponse, StressAnalysisResponse
from services.stress_service import StressService

router = APIRouter(tags=["Stress Prediction"])

@router.post("/stress-predict", response_model=StressPredictResponse)
def predict_stress(
    body: StressPredictRequest,
    _: int = Depends(get_current_user_id),
):
    return StressService.predict_stress(body)

@router.post("/stress-predict-llm", response_model=StressPredictResponse, deprecated=True)
def predict_stress_llm(
    body: StressPredictRequest,
    _: int = Depends(get_current_user_id),
):
    return StressService.predict_stress_llm(body)

@router.post("/stress-predict/explain", response_model=StressExplainResponse)
def explain_stress_prediction(
    body: StressPredictRequest,
    _: int = Depends(get_current_user_id),
    top_n: int = Query(8, ge=1, le=50),
):
    prediction, explanation = StressService.explain_stress(body, top_n)
    return StressExplainResponse(
        stress_level_pct=prediction.stress_level_pct,
        is_stressed=prediction.is_stressed,
        explanation=asdict(explanation),
    )

@router.post("/stress-analysis", response_model=StressAnalysisResponse)
def analyze_stress(
    body: StressPredictRequest,
    _: int = Depends(get_current_user_id),
    top_features: int = Query(6, ge=1, le=20),
    top_k: int = Query(4, ge=1, le=10),
):
    result = StressService.analyze_stress(body, top_features, top_k)
    return StressAnalysisResponse(
        stress_level_pct=result.stress_level_pct,
        is_stressed=result.is_stressed,
        important_features=[asdict(f) for f in result.important_features],
        retrieved_context=[asdict(c) for c in result.retrieved_context],
        explanation=result.explanation,
    )
