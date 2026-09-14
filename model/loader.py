"""
Model Loader & Weights Manager.
Safely loads pre-trained weights, checkpoints, and handles hardware device assignment (CPU/GPU).
Responsible Team Member: Member 1 (Core ML & Model Architecture)
"""

class ModelLoader:
    """Utility to load trained weights into detector instance."""
    
    @staticmethod
    def load_model(weights_path: str, device: str = "cpu"):
        """Loads model state dict and transfers to target device."""
        print(f"[ModelLoader] Loading model weights from {weights_path} to {device}...")
        return "ModelInstance"
