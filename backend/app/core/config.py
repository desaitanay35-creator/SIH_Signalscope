from pathlib import Path
from typing import List
from typing import Any, List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    SignalScope Application Settings.
    
    All settings can be overridden via environment variables or a .env file.
    Note: Model dimensions and normalization parameters are configurable and
    should be matched to the ML team's model specification when provided.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SignalScope API"
    PROJECT_DESCRIPTION: str = "SignalScope — Telling Real From Synthetic in the Age of Generative Media."
    VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"

    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Storage & Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    STORAGE_DIR: Path = Path("storage")
    ORIGINALS_DIR_NAME: str = "originals"
    HEATMAPS_DIR_NAME: str = "heatmaps"

    # Database Configuration (MVP uses SQLite; easy migration to PostgreSQL via DATABASE_URL)
    DATABASE_URL: str = f"sqlite:///{Path(__file__).resolve().parent.parent.parent / 'signalscope.db'}"

    # CORS Configuration
    CORS_ORIGINS: List[str] = ["*"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            if v.strip().startswith("[") and v.strip().endswith("]"):
                import json
                return json.loads(v)
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

    # Image Validation Constraints
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB limit
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    MIN_IMAGE_DIMENSION: int = 32
    MAX_IMAGE_DIMENSION: int = 8192
    MAX_IMAGE_PIXELS: int = 67_108_864  # 8192 * 8192 max pixel count for decompression bomb protection

    # ML Model Configuration (Loosely coupled with ML implementation)
    MODEL_VERSION: str = "signalscope-v1"
    MODEL_WEIGHTS_PATH: str = "model/weights/signalscope_detector.pth"
    DECISION_THRESHOLD: float = 0.50

    # Model Input Dimensions & Normalization
    # NOTE: Fully configurable. Not assuming 224x224 or specific ImageNet stats are final.
    # The ML team specifies these via environment variables or runtime configuration.
    MODEL_INPUT_WIDTH: int = 224
    MODEL_INPUT_HEIGHT: int = 224
    MODEL_NORM_MEAN: List[float] = [0.485, 0.456, 0.406]
    MODEL_NORM_STD: List[float] = [0.229, 0.224, 0.225]

    # Calibration Configuration
    # NOTE: Do NOT invent calibration parameters. Calibration must only be applied
    # when empirical parameters obtained from validation data are available.
    CALIBRATION_ENABLED: bool = False
    CALIBRATION_METHOD: str = "none"  # e.g., 'platt', 'temperature', or 'none'
    CALIBRATION_TEMPERATURE: float = 1.0

    @property
    def originals_storage_path(self) -> Path:
        return self.STORAGE_DIR / self.ORIGINALS_DIR_NAME

    @property
    def heatmaps_storage_path(self) -> Path:
        return self.STORAGE_DIR / self.HEATMAPS_DIR_NAME


settings = Settings()
