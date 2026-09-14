from fastapi import APIRouter
from app.ml.model_loader import ModelLoader
from app.schemas.analysis import ModelInfoResponse

router = APIRouter()


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get active detector metadata",
    responses={
        200: {
            "model": ModelInfoResponse,
            "description": "Metadata for the active detector module, version, task, and loaded state.",
        }
    },
)
async def get_model_info():
    """
    Returns information regarding the active ML detector module and loaded state.
    Loosely coupled: does not expose or hardcode specific internal architectures.
    """
    detector = ModelLoader.get_detector()
    info = detector.get_info()
    return ModelInfoResponse(**info)
