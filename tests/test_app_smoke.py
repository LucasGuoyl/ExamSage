from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_homepage_renders_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EXAMSAGE_DATA_DIR", str(tmp_path / "data"))
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
