"""
Centralized configuration settings for SignalScope.
Handles environment variables, default paths, and system parameters.
Responsible Team Member: Member 6 (MLOps & Config)
"""

import os
from dataclasses import dataclass

@dataclass
class Settings:
    PROJECT_NAME: str = "SignalScope"
    VERSION: str = "1.0.0"
    MODEL_WEIGHTS_PATH: str = os.getenv("MODEL_WEIGHTS_PATH", "weights/signalscope_v1.pth")
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    HOST: str = "0.0.0.0"
    PORT: int = 8000

settings = Settings()
