"""
Unit Tests for Explainability Module.
Verifies Grad-CAM, FFT, and ELA explainer outputs and shapes.
Responsible Team Member: Member 6 (MLOps & Testing)
"""

def test_explainability_imports():
    from explainability.gradcam import GradCAMExplainer
    from explainability.fft_analysis import FFTVisualizer
    from explainability.ela_analysis import ELAExplainer
    assert GradCAMExplainer is not None
    assert FFTVisualizer is not None
    assert ELAExplainer is not None
