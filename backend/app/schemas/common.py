from typing import Any, Optional
from pydantic import BaseModel, Field


class ErrorPayload(BaseModel):
    """Payload details for API errors."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error explanation")
    details: Optional[Any] = Field(None, description="Optional extra error details")


class ErrorResponse(BaseModel):
    """Standardized error envelope."""
    error: ErrorPayload

