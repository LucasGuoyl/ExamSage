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


def test_claimed_read_handle_is_os_read_only(tmp_path):
    filesystem = OwnedArtifactFilesystem()
    root = tmp_path / "data"
    root.mkdir()
    artifact = root / "artifact"
    artifact.write_bytes(b"immutable")
    opened = artifact.stat()

    with filesystem.anchor_directory(root) as anchor:
        with filesystem.open_claimed_file(
            anchor,
            "artifact",
            expected_parent_identity=anchor.identity,
            expected_source_identity=(opened.st_dev, opened.st_ino),
            expected_sha256=_sha256(b"immutable"),
            expected_size=9,
        ) as source:
            with pytest.raises(OSError):
                os.write(source.descriptor, b"changed")

    assert artifact.read_bytes() == b"immutable"


@pytest.mark.parametrize("raise_inside", [False, True])
def test_unreleased_temporary_file_is_cleaned_on_context_exit(tmp_path, raise_inside):
    filesystem = OwnedArtifactFilesystem()
    root = tmp_path / "data"
    root.mkdir()

    with filesystem.anchor_directory(root) as anchor:
        if raise_inside:
            with pytest.raises(RuntimeError, match="before journal"):
                with filesystem.create_temporary_file(
                    anchor,
                    ".owned.tmp",
                    expected_parent_identity=anchor.identity,
                ) as temporary:
                    os.write(temporary.descriptor, b"uncommitted")
                    raise RuntimeError("before journal")
        else:
            with filesystem.create_temporary_file(
                anchor,
                ".owned.tmp",
                expected_parent_identity=anchor.identity,
            ) as temporary:
                os.write(temporary.descriptor, b"uncommitted")

    assert not (root / ".owned.tmp").exists()


def test_exclusive_child_creation_refuses_an_existing_directory(tmp_path):
    filesystem = OwnedArtifactFilesystem()
    root = tmp_path / "data"
    (root / "evidence").mkdir(parents=True)

    with filesystem.anchor_directory(root) as anchor:
        with pytest.raises(OwnedFilesystemError) as caught:
            with filesystem.create_new_child_directory(
                anchor,
                "evidence",
                expected_parent_identity=anchor.identity,
            ):
                pytest.fail("an existing unregistered directory cannot be adopted")

    assert caught.value.code == "owned_destination_exists"


def test_fixed_mutation_file_is_created_then_reopened_with_one_identity(tmp_path):
    filesystem = OwnedArtifactFilesystem()
    root = tmp_path / "data"
    root.mkdir()

    with filesystem.anchor_directory(root) as anchor:
        with filesystem.open_or_create_mutation_file(
            anchor,
            ".registry.sqlite3",
            expected_parent_identity=anchor.identity,
        ) as created:
            os.write(created.descriptor, b"registry")
            os.fsync(created.descriptor)
            identity = created.identity
        with filesystem.open_or_create_mutation_file(
            anchor,
            ".registry.sqlite3",
            expected_parent_identity=anchor.identity,
        ) as reopened:
            assert reopened.identity == identity
            assert filesystem.hash_open_file(reopened) == (_sha256(b"registry"), 8)


def test_open_existing_mutation_file_never_creates_a_missing_authority_name(tmp_path):
    filesystem = OwnedArtifactFilesystem()
    root = tmp_path / "data"
    root.mkdir()

    with filesystem.anchor_directory(root) as anchor:
        with pytest.raises(OwnedFilesystemError) as caught:
            with filesystem.open_mutation_file(
                anchor,
                ".registry.log",
                expected_parent_identity=anchor.identity,
            ):
                pytest.fail("a missing authority file must not be created")

    assert caught.value.code == "owned_not_found"
    assert not (root / ".registry.log").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound temp cleanup")
def test_windows_temp_cleanup_marks_exact_handle_even_when_directory_flush_fails(
    tmp_path,
    monkeypatch,
):
    filesystem = OwnedArtifactFilesystem()
    root = tmp_path / "data"
    root.mkdir()

    def fail_flush(parent):
        del parent
        raise OSError("directory flush failed")

    monkeypatch.setattr(filesystem, "_flush_directory_windows", fail_flush)
    with filesystem.anchor_directory(root) as anchor:
        with pytest.raises(OwnedFilesystemError):
            with filesystem.create_temporary_file(
                anchor,
                ".owned.tmp",
                expected_parent_identity=anchor.identity,
            ) as temporary:
                os.write(temporary.descriptor, b"uncommitted")

    assert list(root.iterdir()) == []


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
    tombstones = [path for path in root.iterdir() if path.name.startswith(".owned-directory-")]
    assert len(tombstones) == 1
    assert tombstones[0].is_dir()


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX name-race hardening")
def test_posix_replace_quarantines_a_last_moment_source_substitution(tmp_path):
    class SubstitutingFilesystem(OwnedArtifactFilesystem):
        def __init__(self):
            super().__init__()
            self.substituted = False

        def _rename_noreplace_posix(self, parent, source_name, destination_name):
            if not self.substituted and destination_name == "artifact":
                self.substituted = True
                (parent.path / source_name).rename(parent.path / ".claimed-displaced")
                (parent.path / source_name).write_bytes(b"attacker")
            return super()._rename_noreplace_posix(parent, source_name, destination_name)

    filesystem = SubstitutingFilesystem()
    root = tmp_path / "data"
    root.mkdir()
    with filesystem.anchor_directory(root) as anchor:
        with filesystem.create_temporary_file(
            anchor,
            ".owned.tmp",
            expected_parent_identity=anchor.identity,
        ) as temporary:
            os.write(temporary.descriptor, b"claimed")
            os.fsync(temporary.descriptor)
            with pytest.raises(OwnedFilesystemError) as caught:
                filesystem.replace_open_file(
                    anchor,
                    temporary,
                    ".owned.tmp",
                    "artifact",
                    expected_parent_identity=anchor.identity,
                    expected_source_identity=temporary.identity,
                    expected_sha256=_sha256(b"claimed"),
                    expected_size=7,
                    replace_existing=False,
                )

    assert caught.value.code == "owned_identity_changed"
    assert not (root / "artifact").exists()
    assert b"attacker" in [path.read_bytes() for path in root.iterdir() if path.is_file()]


@pytest.mark.skipif(os.name == "nt", reason="POSIX name-race hardening")
def test_posix_empty_directory_removal_never_deletes_a_substitute(tmp_path):
    class SubstitutingFilesystem(OwnedArtifactFilesystem):
        def __init__(self):
            super().__init__()
            self.substituted = False

        def before_mutation(self, operation, parent, source_name):
            if operation == "remove_directory" and not self.substituted:
                self.substituted = True
                (parent.path / source_name).rename(parent.path / ".claimed-directory-displaced")
                (parent.path / source_name).mkdir()

    filesystem = SubstitutingFilesystem()
    root = tmp_path / "data"
    child = root / "parts"
    child.mkdir(parents=True)
    child_stat = child.stat(follow_symlinks=False)
    with filesystem.anchor_directory(root) as anchor:
        with pytest.raises(OwnedFilesystemError) as caught:
            filesystem.remove_empty_directory(
                anchor,
                "parts",
                expected_parent_identity=anchor.identity,
                expected_child_identity=(child_stat.st_dev, child_stat.st_ino),
            )

    assert caught.value.code == "owned_identity_changed"
    assert (root / ".claimed-directory-displaced").exists()
    assert any(path.is_dir() for path in root.iterdir() if path.name.startswith(".owned-directory-"))


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


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound quarantine")
def test_windows_claimed_delete_renames_the_verified_handle(tmp_path, monkeypatch):
    filesystem = OwnedArtifactFilesystem()
    path_moves: list[int] = []
    handle_moves: list[str] = []
    real_path_move = filesystem._move_file_ex_windows
    real_handle_move = filesystem._rename_handle_windows

    def recording_path_move(source, destination, flags):
        path_moves.append(flags)
        return real_path_move(source, destination, flags)

    def recording_handle_move(handle, destination):
        handle_moves.append(destination.name)
        return real_handle_move(handle, destination)

    monkeypatch.setattr(filesystem, "_move_file_ex_windows", recording_path_move)
    monkeypatch.setattr(filesystem, "_rename_handle_windows", recording_handle_move)
    root = tmp_path / "data"
    root.mkdir()
    artifact = root / "artifact"
    artifact.write_bytes(b"claimed")
    artifact_stat = artifact.stat()
    with filesystem.anchor_directory(root) as anchor:
        filesystem.delete_claimed_file(
            anchor,
            "artifact",
            expected_parent_identity=anchor.identity,
            expected_source_identity=(artifact_stat.st_dev, artifact_stat.st_ino),
            expected_sha256=_sha256(b"claimed"),
            expected_size=7,
        )

    assert path_moves == []
    assert len(handle_moves) == 1
    assert handle_moves[0].startswith(".owned-quarantine-")
