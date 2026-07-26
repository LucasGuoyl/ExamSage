from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_homepage_renders_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EXAMSAGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("EXAMSAGE_AGENT_V2", raising=False)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()

    assert not app.exception
    assert app.title[0].value == "🎓 ExamSage"
    assert app.radio[0].options == [
        "OpenAI",
        "Google Gemini",
        "Custom OpenAI-compatible (experimental)",
    ]
    button_labels = [button.label for button in app.button]
    assert "Estimate cost" in button_labels
    assert "Build my ExamSage agent" not in button_labels
    assert "Choose course folder" not in button_labels


def test_agent_route_fails_safely_when_worker_is_unavailable(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EXAMSAGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EXAMSAGE_AGENT_V2", "1")
    monkeypatch.setenv("EXAMSAGE_WORKER_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("EXAMSAGE_WORKER_TOKEN", "test-token")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()

    assert not app.exception
    assert app.title[0].value == "🎓 ExamSage"
    assert any("Worker unavailable" in item.value for item in app.error)
    button_labels = [button.label for button in app.button]
    assert "Estimate cost" not in button_labels
    assert "Build my ExamSage agent" not in button_labels
