import os

def test_model_exists():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "model.pkl")
    assert os.path.exists(model_path)
