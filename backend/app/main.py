from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.api.routes import health, model, analysis
from app.core.config import settings
from app.core.errors import AppException
from app.core.logging import logger
from app.db.database import init_db
from app.schemas.common import ErrorResponse, ErrorPayload


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    logger.info(f"{settings.PROJECT_NAME} v{settings.VERSION} starting up.")
    init_db()
    logger.info("Database initialized.")
    yield
    logger.info(f"{settings.PROJECT_NAME} shutting down.")


def create_app() -> FastAPI:
    """Application factory for SignalScope FastAPI backend."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Cross-Origin Resource Sharing (CORS) Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1. Custom Domain Exception Handler (AppException)
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.error(f"AppException [{exc.code}]: {exc.message} (path={request.url.path})")
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorPayload(
                    code=exc.code,
                    message=exc.message,
                    details=exc.details
                )
            ).model_dump()
        )

    # 2. FastAPI Request Validation Exception Handler (e.g. missing multipart fields, query validation)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Request validation error on path {request.url.path}: {exc.errors()}")
        sanitized_errors = [
            {
                "field": ".".join(str(loc) for loc in err.get("loc", []) if loc != "body"),
                "message": err.get("msg", ""),
                "type": err.get("type", "")
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorPayload(
                    code="VALIDATION_ERROR",
                    message="The request payload or parameters failed validation.",
                    details=sanitized_errors
                )
            ).model_dump()
        )

    # 3. Starlette HTTP Exception Handler (e.g. standard 404, 405)
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(f"HTTP exception [{exc.status_code}] on path {request.url.path}: {exc.detail}")
        code_map = {
            400: "BAD_REQUEST",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "FILE_TOO_LARGE",
            500: "INTERNAL_SERVER_ERROR",
            503: "SERVICE_UNAVAILABLE"
        }
        code = code_map.get(exc.status_code, f"HTTP_{exc.status_code}")
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorPayload(
                    code=code,
                    message=str(exc.detail),
                    details=None
                )
            ).model_dump()
        )

    # 4. General Unhandled Exception Handler (masks internal stack traces and paths)
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled Server Error on path {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorPayload(
                    code="INTERNAL_SERVER_ERROR",
                    message="An internal server error occurred while processing the request.",
                    details=None
                )
            ).model_dump()
        )

    # Include Route Modules under /api/v1
    app.include_router(health.router, prefix=settings.API_V1_STR)
    app.include_router(model.router, prefix=settings.API_V1_STR)
    app.include_router(analysis.router, prefix=settings.API_V1_STR)

    return app


app = create_app()
