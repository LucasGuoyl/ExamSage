from __future__ import annotations

import secrets

import pytest

import exam_predictor.runtime.credential_vault as vault_module
from exam_predictor.runtime.credential_vault import (
    KeyringCredentialVault,
    VaultUnavailableError,
)


class FakeKeyring:
    def __init__(self) -> None:
        self.passwords: dict[tuple[str, str], str] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.failure: Exception | None = None

    def set_password(self, service: str, account: str, password: str) -> None:
        self.calls.append(("set", service, account))
        if self.failure is not None:
            raise self.failure
        self.passwords[(service, account)] = password

    def get_password(self, service: str, account: str) -> str | None:
        self.calls.append(("get", service, account))
        if self.failure is not None:
            raise self.failure
        return self.passwords.get((service, account))

    def delete_password(self, service: str, account: str) -> None:
        self.calls.append(("delete", service, account))
        if self.failure is not None:
            raise self.failure
        self.passwords.pop((service, account), None)


def test_vault_saves_replaces_loads_checks_and_deletes_one_provider_secret():
    backend = FakeKeyring()
    vault = KeyringCredentialVault(backend)

    assert vault.load("primary") is None
    assert vault.exists("primary") is False
    vault.save("primary", "first-secret")
    vault.save("primary", "replacement-secret")

    assert vault.load("primary") == "replacement-secret"
    assert vault.exists("primary") is True
    assert backend.passwords == {("ExamSage", "provider:primary"): "replacement-secret"}

    vault.delete("primary")
    vault.delete("primary")
    assert vault.load("primary") is None


@pytest.mark.parametrize(
    "profile_id",
    ["", " profile", "profile/name", "a" * 65, "provider:primary"],
)
def test_vault_rejects_invalid_profile_ids_before_backend_access(profile_id: str):
    backend = FakeKeyring()
    vault = KeyringCredentialVault(backend)

    with pytest.raises(ValueError, match="Invalid provider profile ID"):
        vault.exists(profile_id)

    assert backend.calls == []


@pytest.mark.parametrize("operation", ["save", "load", "exists", "delete"])
def test_vault_discards_backend_exception_and_secret_from_the_exception_chain(
    operation: str,
):
    sentinel = "vault-secret-" + secrets.token_hex(16)
    backend = FakeKeyring()
    backend.failure = RuntimeError(f"backend rejected {sentinel}")
    vault = KeyringCredentialVault(backend)

    with pytest.raises(
        VaultUnavailableError,
        match="^Secure credential storage is unavailable$",
    ) as captured:
        if operation == "save":
            vault.save("primary", sentinel)
        else:
            getattr(vault, operation)("primary")

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in repr(captured.value)


def test_missing_native_keyring_import_is_reported_without_exception_context(
    monkeypatch: pytest.MonkeyPatch,
):
    sentinel = "native-import-secret-path"

    def fail_import(_name: str):
        raise ImportError(sentinel)

    monkeypatch.setattr(vault_module, "import_module", fail_import)

    with pytest.raises(VaultUnavailableError) as captured:
        KeyringCredentialVault()

    assert str(captured.value) == "Secure credential storage is unavailable"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in repr(captured.value)


def test_native_missing_password_delete_is_idempotent():
    class PasswordDeleteError(Exception):
        pass

    PasswordDeleteError.__module__ = "keyring.errors"

    class MissingDeleteKeyring(FakeKeyring):
        def delete_password(self, service: str, account: str) -> None:
            self.calls.append(("delete", service, account))
            raise PasswordDeleteError("Password not found")

    backend = MissingDeleteKeyring()
    vault = KeyringCredentialVault(backend)

    vault.delete("primary")

    assert backend.calls == [("delete", "ExamSage", "provider:primary")]
