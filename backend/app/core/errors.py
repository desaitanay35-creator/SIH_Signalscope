from typing import Any, Optional


class AppException(Exception):
    """Base exception for SignalScope application errors."""
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: Optional[Any] = None
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class InvalidImageError(AppException):
    """Raised when an uploaded file cannot be decoded as an image or has invalid dimensions."""
    def __init__(self, message: str = "The uploaded file is not a valid image.", details: Optional[Any] = None):
        super().__init__(
            code="INVALID_IMAGE",
            message=message,
            status_code=400,
            details=details
        )


class FileTooLargeError(AppException):
    """Raised when an uploaded file exceeds the configured maximum upload size."""
    def __init__(self, message: str = "The uploaded file exceeds the maximum allowed size.", details: Optional[Any] = None):
        super().__init__(
            code="FILE_TOO_LARGE",
            message=message,
            status_code=413,
            details=details
        )


class UnsupportedImageTypeError(AppException):
    """Raised when an uploaded file format is not supported."""
    def __init__(self, message: str = "The image format is not supported. Allowed formats: JPEG, PNG, WEBP.", details: Optional[Any] = None):
        super().__init__(
            code="UNSUPPORTED_IMAGE_TYPE",
            message=message,
            status_code=400,
            details=details
        )


class ModelUnavailableError(AppException):
    """Raised when model weights or ML inference service is not available."""
    def __init__(
        self,
        message: str = "ML detector is unavailable or model weights are not loaded.",
        details: Optional[Any] = None
    ):
        super().__init__(
            code="MODEL_UNAVAILABLE",
            message=message,
            status_code=503,
            details=details
        )


class ModelIncompatibleError(AppException):
    """Raised when model weights exist on disk but the checkpoint is corrupt,
    unreadable, or declares an architecture/state_dict incompatible with the
    detector attempting to load it.

    Deliberately distinct from ModelUnavailableError: a missing-weights file
    is an expected/transient condition (HTTP 503, backend falls back to
    StubImageDetector); a present-but-broken checkpoint is a real bug that
    must fail loudly at startup, never silently substitute a stub detector.
    """
    def __init__(
        self,
        message: str = "The configured model checkpoint is corrupt or incompatible with the expected architecture.",
        details: Optional[Any] = None
    ):
        super().__init__(
            code="MODEL_INCOMPATIBLE",
            message=message,
            status_code=500,
            details=details
        )


class InvalidModelOutputError(AppException):
    """Raised when the detector returns invalid, non-numeric, or out-of-range output."""
    def __init__(
        self,
        message: str = "The detector returned an invalid output format or invalid probabilities.",
        details: Optional[Any] = None
    ):
        super().__init__(
            code="INVALID_MODEL_OUTPUT",
            message=message,
            status_code=500,
            details=details
        )


class InferenceError(AppException):
    """Raised when an unexpected error occurs during model execution."""
    def __init__(
        self,
        message: str = "An error occurred during model inference.",
        details: Optional[Any] = None
    ):
        super().__init__(
            code="INFERENCE_ERROR",
            message=message,
            status_code=500,
            details=details
        )


class AnalysisNotFoundError(AppException):
    """Raised when an analysis ID is not found in storage/database."""
    def __init__(self, message: str = "Analysis record not found.", details: Optional[Any] = None):
        super().__init__(
            code="ANALYSIS_NOT_FOUND",
            message=message,
            status_code=404,
            details=details
        )


class AnalysisFailedError(AppException):
    """Raised when the analysis pipeline encounters an unrecoverable failure."""
    def __init__(self, message: str = "Analysis pipeline failed to process the request.", details: Optional[Any] = None):
        super().__init__(
            code="ANALYSIS_FAILED",
            message=message,
            status_code=500,
            details=details
        )


class InvalidAnalysisIdError(AppException):
    """Raised when an analysis ID is not a valid UUID format."""
    def __init__(self, message: str = "The provided analysis ID is not a valid UUID format.", details: Optional[Any] = None):
        super().__init__(
            code="INVALID_ANALYSIS_ID",
            message=message,
            status_code=400,
            details=details
        )


class DatabaseError(AppException):
    """Raised when a database query or transaction operation fails."""
    def __init__(self, message: str = "A database error occurred while processing the request.", details: Optional[Any] = None):
        super().__init__(
            code="DATABASE_ERROR",
            message=message,
            status_code=500,
            details=details
        )

