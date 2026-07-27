from __future__ import annotations

import hashlib
import os

import pytest

from exam_predictor.workspace.filesystem import (
    OwnedArtifactFilesystem,
    OwnedFilesystemError,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_trusted_root_anchor_creates_child_and_publishes_live_temp(tmp_path):
    filesystem = OwnedArtifactFilesystem()
    root = tmp_path / "data"
    root.mkdir()
    content = b"identity-bound bytes"

    with filesystem.anchor_directory(root) as root_anchor:
        assert root_anchor.identity == filesystem.anchor_identity(root_anchor)
        with filesystem.create_child_directory(
            root_anchor,
            "parts",
            expected_parent_identity=root_anchor.identity,
        ) as parts_anchor:
            with filesystem.create_temporary_file(
                parts_anchor,
                ".owned.tmp",
                expected_parent_identity=parts_anchor.identity,
            ) as temporary:
                os.write(temporary.descriptor, content)
                os.fsync(temporary.descriptor)
                assert filesystem.hash_open_file(temporary) == (
                    _sha256(content),
                    len(content),
                )
                result = filesystem.replace_open_file(
                    parts_anchor,
                    temporary,
                    ".owned.tmp",
                    "artifact",
                    expected_parent_identity=parts_anchor.identity,
                    expected_source_identity=temporary.identity,
                    expected_sha256=_sha256(content),
                    expected_size=len(content),
                    replace_existing=True,
                )

            assert result.identity == temporary.identity
            assert result.sha256 == _sha256(content)
            assert (root / "parts" / "artifact").read_bytes() == content


def test_missing_or_linked_trusted_root_is_rejected_before_write(tmp_path):
    filesystem = OwnedArtifactFilesystem()
    missing = tmp_path / "missing"
    with pytest.raises(OwnedFilesystemError) as caught:
        with filesystem.anchor_directory(missing):
            pytest.fail("a missing trusted root cannot be bootstrapped here")
    assert caught.value.code == "owned_root_missing"
    assert not missing.exists()

    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory links unavailable: {error}")
    with pytest.raises(OwnedFilesystemError):
        with filesystem.anchor_directory(linked):
            pytest.fail("trusted roots cannot be links")


def test_mutation_rejects_wrong_parent_identity(tmp_path):
    filesystem = OwnedArtifactFilesystem()
    root = tmp_path / "data"
    root.mkdir()

    with filesystem.anchor_directory(root) as anchor:
        with filesystem.create_temporary_file(
            anchor,
            ".owned.tmp",
            expected_parent_identity=anchor.identity,
        ) as temporary:
            os.write(temporary.descriptor, b"new")
            os.fsync(temporary.descriptor)
            with pytest.raises(OwnedFilesystemError) as caught:
                filesystem.replace_open_file(
                    anchor,
                    temporary,
                    ".owned.tmp",
                    "artifact",
                    expected_parent_identity=(0, 0),
                    expected_source_identity=temporary.identity,
                    expected_sha256=_sha256(b"new"),
                    expected_size=3,
                    replace_existing=True,
                )

    assert caught.value.code == "owned_identity_changed"
    assert not (root / "artifact").exists()


def test_existing_child_read_and_empty_directory_removal_stay_anchored(tmp_path):
    filesystem = OwnedArtifactFilesystem()
    root = tmp_path / "data"
    child = root / "parts"
    child.mkdir(parents=True)
    artifact = child / "artifact"
    artifact.write_bytes(b"claimed")

    with filesystem.anchor_directory(root) as root_anchor:
        with filesystem.anchor_child_directory(
            root_anchor,
            "parts",
            expected_parent_identity=root_anchor.identity,
        ) as parts_anchor:
            opened_identity, content = filesystem.read_named_file(
                parts_anchor,
                "artifact",
                expected_parent_identity=parts_anchor.identity,
                maximum_bytes=32,
            )
            assert opened_identity == (
                artifact.stat().st_dev,
                artifact.stat().st_ino,
            )
            assert content == b"claimed"
            filesystem.delete_claimed_file(
                parts_anchor,
                "artifact",
                expected_parent_identity=parts_anchor.identity,
                expected_source_identity=opened_identity,
                expected_sha256=_sha256(content),
                expected_size=len(content),
            )
        filesystem.remove_empty_directory(
            root_anchor,
            "parts",
            expected_parent_identity=root_anchor.identity,
            expected_child_identity=parts_anchor.identity,
        )

    assert not child.exists()


def test_claimed_delete_quarantines_and_never_deletes_a_substitution(tmp_path):
    filesystem = OwnedArtifactFilesystem()
    root = tmp_path / "data"
    root.mkdir()
    claimed = root / "artifact"

    with filesystem.anchor_directory(root) as anchor:
        claimed.write_bytes(b"claimed")
        claimed_stat = claimed.stat(follow_symlinks=False)
        claimed_identity = (claimed_stat.st_dev, claimed_stat.st_ino)
        filesystem.delete_claimed_file(
            anchor,
            "artifact",
            expected_parent_identity=anchor.identity,
            expected_source_identity=claimed_identity,
            expected_sha256=_sha256(b"claimed"),
            expected_size=7,
        )
        assert not claimed.exists()

        claimed.write_bytes(b"replacement")
        with pytest.raises(OwnedFilesystemError) as caught:
            filesystem.delete_claimed_file(
                anchor,
                "artifact",
                expected_parent_identity=anchor.identity,
                expected_source_identity=claimed_identity,
                expected_sha256=_sha256(b"claimed"),
                expected_size=7,
            )

    assert caught.value.code == "owned_identity_changed"
    assert claimed.read_bytes() == b"replacement"


@pytest.mark.skipif(os.name != "nt", reason="Windows durability primitives")
def test_windows_replace_uses_documented_write_through_and_required_flush(tmp_path, monkeypatch):
    filesystem = OwnedArtifactFilesystem()
    calls: list[int] = []
    real_move = filesystem._move_file_ex_windows

    def recording_move(source, destination, flags):
        calls.append(flags)
        return real_move(source, destination, flags)

    monkeypatch.setattr(filesystem, "_move_file_ex_windows", recording_move)
    root = tmp_path / "data"
    root.mkdir()
    with filesystem.anchor_directory(root) as anchor:
        with filesystem.create_temporary_file(
            anchor,
            ".owned.tmp",
            expected_parent_identity=anchor.identity,
        ) as temporary:
            os.write(temporary.descriptor, b"durable")
            os.fsync(temporary.descriptor)
            result = filesystem.replace_open_file(
                anchor,
                temporary,
                ".owned.tmp",
                "artifact",
                expected_parent_identity=anchor.identity,
                expected_source_identity=temporary.identity,
                expected_sha256=_sha256(b"durable"),
                expected_size=7,
                replace_existing=True,
            )

    assert calls == [filesystem.MOVEFILE_REPLACE_EXISTING | filesystem.MOVEFILE_WRITE_THROUGH]
    assert result.rename_write_through is True
    assert result.final_file_flushed is True
