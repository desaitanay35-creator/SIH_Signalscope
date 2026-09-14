import os
import uuid
from pathlib import Path
from typing import Optional, Tuple
from app.core.config import settings
from app.core.logging import logger


class StorageService:
    """
    Manages filesystem storage for uploaded images and forensic heatmaps.
    
    Security design:
    - Never uses client-provided filenames as filesystem paths.
    - Uses random UUIDs for all stored files to avoid path traversal and collisions.
    - Returns relative paths or IDs, never exposing server internal directory structures.
    - Supports safe transactional cleanup for failed pipeline runs.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or settings.BASE_DIR
        self.originals_dir = self.base_dir / settings.originals_storage_path
        self.heatmaps_dir = self.base_dir / settings.heatmaps_storage_path
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensures that required storage directories exist."""
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.heatmaps_dir.mkdir(parents=True, exist_ok=True)

    def save_original(self, file_bytes: bytes, extension: str) -> Tuple[str, str]:
        """
        Saves original image bytes under storage/originals/<file_uuid>.<clean_ext>.
        
        Returns:
            Tuple of (file_uuid, relative_storage_path)
        """
        file_uuid = str(uuid.uuid4())
        clean_ext = extension.lstrip(".").lower()
        if not clean_ext:
            clean_ext = "jpg"
            
        filename = f"{file_uuid}.{clean_ext}"
        target_path = self.originals_dir / filename

        with open(target_path, "wb") as f:
            f.write(file_bytes)

        relative_path = os.path.relpath(target_path, self.base_dir).replace("\\", "/")
        logger.info(f"Saved original image ({len(file_bytes)} bytes) to relative path '{relative_path}'")
        return file_uuid, relative_path

    def delete_original(self, relative_path: str) -> bool:
        """
        Safely deletes an original image created during an analysis transaction.
        
        Security & Safety guarantees:
        - Resolves relative_path against base_dir.
        - Strictly verifies that the resolved path is located within originals_dir.
        - Strictly verifies that it is a file and not a directory.
        - Prevents directory traversal attacks.
        - Only deletes if the file exists.
        """
        if not relative_path:
            return False
        try:
            target_path = (self.base_dir / relative_path).resolve()
            originals_dir_resolved = self.originals_dir.resolve()
            
            # Verify path containment strictly within originals directory
            if originals_dir_resolved not in target_path.parents:
                logger.warning(f"Refusing to delete path outside originals directory: {target_path}")
                return False
                
            if target_path.exists() and target_path.is_file():
                target_path.unlink()
                logger.info(f"Cleaned up transaction image: {target_path.name}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete stored original image '{relative_path}': {str(e)}")
        return False

    def save_heatmap(self, analysis_id: str, image_bytes: bytes) -> str:
        """
        Saves generated Grad-CAM heatmap under storage/heatmaps/<analysis_id>.jpg.
        
        Returns:
            Relative storage path
        """
        filename = f"{analysis_id}.jpg"
        target_path = self.heatmaps_dir / filename

        with open(target_path, "wb") as f:
            f.write(image_bytes)

        relative_path = os.path.relpath(target_path, self.base_dir).replace("\\", "/")
        logger.info(f"Saved heatmap for analysis {analysis_id} to '{relative_path}'")
        return relative_path

    def delete_heatmap(self, analysis_id: str) -> bool:
        """
        Safely deletes a heatmap image created during a failed transaction.
        """
        if not analysis_id:
            return False
        try:
            target_path = (self.heatmaps_dir / f"{analysis_id}.jpg").resolve()
            heatmaps_dir_resolved = self.heatmaps_dir.resolve()
            
            if heatmaps_dir_resolved not in target_path.parents:
                logger.warning(f"Refusing to delete path outside heatmaps directory: {target_path}")
                return False
                
            if target_path.exists() and target_path.is_file():
                target_path.unlink()
                logger.info(f"Cleaned up transaction heatmap: {target_path.name}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete heatmap for analysis '{analysis_id}': {str(e)}")
        return False

    def get_heatmap_path(self, analysis_id: str) -> Optional[Path]:
        """
        Retrieves the absolute path of a heatmap if it exists on disk.
        Ensures strict path containment within heatmaps_dir to prevent directory traversal.
        """
        if not analysis_id:
            return None
        try:
            target_path = (self.heatmaps_dir / f"{analysis_id}.jpg").resolve()
            heatmaps_dir_resolved = self.heatmaps_dir.resolve()
            if heatmaps_dir_resolved in target_path.parents and target_path.exists() and target_path.is_file():
                return target_path
        except Exception as e:
            logger.error(f"Error resolving heatmap path for '{analysis_id}': {str(e)}")
        return None

    def get_original_path(self, relative_path: str) -> Optional[Path]:
        """
        Resolves a relative storage path safely against base directory.
        """
        target_path = (self.base_dir / relative_path).resolve()
        # Prevent directory traversal
        if self.originals_dir.resolve() in target_path.parents and target_path.exists():
            return target_path
        return None
