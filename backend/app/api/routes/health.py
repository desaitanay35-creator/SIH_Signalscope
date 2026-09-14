from fastapi import APIRouter
from app.ml.model_loader import ModelLoader
from app.schemas.analysis import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check service health and model availability",
    responses={
        200: {
            "model": HealthResponse,
            "description": "System is healthy. model_loaded indicates if ML detector weights are active.",
        }
    },
)
async def health_check():
    """
    Returns system operational status and model readiness.
    Used by load balancers, container orchestrators, and monitoring services.
    """
    detector = ModelLoader.get_detector()
    return HealthResponse(
        status="ok",
        model_loaded=detector.is_loaded()
    )
