from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY, classify_format


def test_supported_formats_are_classified_without_reading_content():
    assert classify_format("lecture.PDF") == "pdf"
    assert classify_format("questions.xlsx") == "spreadsheet"
    assert classify_format("scan.tiff") == "image"
    assert classify_format("bundle.zip") == "archive"


def test_audio_video_executables_and_unknown_files_are_visible_exclusions():
    for name in ("lecture.mp4", "recording.mp3", "setup.exe", "blob.unknown"):
        assert classify_format(name) is None


def test_default_policy_is_bounded_and_versioned():
    assert DEFAULT_SCAN_POLICY.max_workspace_bytes == 1_073_741_824
    assert DEFAULT_SCAN_POLICY.hash_chunk_bytes == 1_048_576
    assert DEFAULT_SCAN_POLICY.policy_version == "workspace-v1"
