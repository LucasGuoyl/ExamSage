from __future__ import annotations

import pytest

from exam_predictor.ui.i18n import (
    SUPPORTED_UI_LANGUAGES,
    UI_LANGUAGE_KEY,
    get_ui_language,
    set_ui_language,
    text,
)


def test_every_ui_message_has_nonempty_english_and_simplified_chinese_copy():
    assert SUPPORTED_UI_LANGUAGES == ("en", "zh-CN")
    english_keys = set(text.catalog("en"))
    chinese_keys = set(text.catalog("zh-CN"))

    assert english_keys == chinese_keys
    assert english_keys
    assert all(value.strip() for value in text.catalog("en").values())
    assert all(value.strip() for value in text.catalog("zh-CN").values())


def test_language_selection_persists_without_touching_academic_artifacts():
    stored_artifact = {"snapshot_id": "snapshot-1", "title": "Limits"}
    state = {"academic_artifact": stored_artifact}

    assert get_ui_language(state) == "en"
    set_ui_language(state, "zh-CN")

    assert get_ui_language(state) == "zh-CN"
    assert state[UI_LANGUAGE_KEY] == "zh-CN"
    assert state["academic_artifact"] is stored_artifact
    assert stored_artifact == {"snapshot_id": "snapshot-1", "title": "Limits"}


def test_invalid_language_is_rejected_instead_of_silently_changing_copy():
    with pytest.raises(ValueError, match="supported"):
        set_ui_language({}, "fr")


def test_translated_copy_formats_named_values_in_both_languages():
    assert text("coverage_banner", "en", covered=2, total=5) == (
        "Initial study map: 2 of 5 sources covered."
    )
    assert text("coverage_banner", "zh-CN", covered=2, total=5) == (
        "初始学习图谱：已覆盖 2/5 个来源。"
    )
