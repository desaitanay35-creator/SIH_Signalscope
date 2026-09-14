"""
I/O & File Management Utilities.
Handles safe file loading, image type validation, and temporary directory management.
Responsible Team Member: Member 6 (MLOps & Infrastructure)
"""

import os
from typing import List

class IOHandler:
    """Utility class for safe file read/write operations."""
    
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]

    @classmethod
    def is_valid_image(cls, filepath: str) -> bool:
        """Validates if file exists and has supported image extension."""
        ext = os.path.splitext(filepath)[1].lower()
        return os.path.isfile(filepath) and ext in cls.ALLOWED_EXTENSIONS
