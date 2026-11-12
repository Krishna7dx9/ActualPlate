import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_vision_worker_warmup_sources_are_syntactically_valid_and_registered():
    api_path = ROOT / "vision_service" / "inference" / "api.py"
    models_path = ROOT / "vision_service" / "inference" / "models.py"
    api_source = api_path.read_text(encoding="utf-8")
    models_source = models_path.read_text(encoding="utf-8")

    ast.parse(api_source)
    ast.parse(models_source)

    assert '@app.post("/warmup")' in api_source
    assert "_check_auth(x_api_token)" in api_source
    assert "get_grounding_dino()" in api_source
    assert "get_sam2_predictor()" in api_source
    assert "get_moge()" in api_source
    assert "_MODEL_LOAD_LOCK" in models_source
    assert "await run_in_threadpool" in api_source
    assert '"volume_cm3": None' in api_source
    assert '"experimental_volume_cm3": None' in api_source
    assert '"volume_status": "unavailable"' in api_source
    assert "from vision_service.settings import settings" in models_source
    assert "MoGe-2 supplies monocular geometry/depth" in models_source
