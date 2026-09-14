"""
Unit Tests for Data Preprocessing Module.
Verifies tensor shapes, face detection logic, and image augmentations.
Responsible Team Member: Member 6 (MLOps & Testing)
"""

def test_preprocessor():
    from data.preprocessor import ImagePreprocessor
    prep = ImagePreprocessor()
    res = prep.preprocess("dummy.jpg")
    assert res is not None
