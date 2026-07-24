from __future__ import annotations

import re
from importlib import import_module
from typing import Protocol, cast

class VaultUnavailableError(RuntimeError):
    """Raised when the operating-system credential store cannot be used."""


class CredentialVault(Protocol):
    def save(self, profile_id: str, api_key: str) -> None:
        """Persist one provider secret outside application files."""

    def load(self, profile_id: str) -> str | None:
        """Return the secret or None without logging it."""

    def exists(self, profile_id: str) -> bool:
        """Report whether a credential exists."""

    def delete(self, profile_id: str) -> None:
        """Remove the vault secret idempotently."""


class KeyringBackend(Protocol):
    def set_password(self, service: str, account: str, password: str) -> None:
        """Store one password."""

    def get_password(self, service: str, account: str) -> str | None:
        """Load one password."""

    def delete_password(self, service: str, account: str) -> None:
        """Delete one password."""


class KeyringCredentialVault:
    SERVICE_NAME = "ExamSage"
    _PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    _UNAVAILABLE = "Secure credential storage is unavailable"

    def __init__(self, backend: KeyringBackend | None = None) -> None:
        unavailable = False
        imported_backend: KeyringBackend | None = None
        if backend is None:
            try:
                imported_backend = cast(KeyringBackend, import_module("keyring"))
            except Exception:
                unavailable = True
        if unavailable:
            raise VaultUnavailableError(self._UNAVAILABLE)
        self._backend = backend or cast(KeyringBackend, imported_backend)

    @classmethod
    def _account(cls, profile_id: str) -> str:
        if cls._PROFILE_ID.fullmatch(profile_id) is None:
            raise ValueError("Invalid provider profile ID for secure credential storage.")
        return f"provider:{profile_id}"

    def save(self, profile_id: str, api_key: str) -> None:
        account = self._account(profile_id)
        unavailable = False
        try:
            self._backend.set_password(self.SERVICE_NAME, account, api_key)
        except Exception:
            unavailable = True
        if unavailable:
            raise VaultUnavailableError(self._UNAVAILABLE)

    def load(self, profile_id: str) -> str | None:
        account = self._account(profile_id)
        unavailable = False
        value: str | None = None
        try:
            value = self._backend.get_password(self.SERVICE_NAME, account)
        except Exception:
            unavailable = True
        if unavailable:
            raise VaultUnavailableError(self._UNAVAILABLE)
        return value

    def exists(self, profile_id: str) -> bool:
        return self.load(profile_id) is not None

    def delete(self, profile_id: str) -> None:
        account = self._account(profile_id)
        unavailable = False
        try:
            self._backend.delete_password(self.SERVICE_NAME, account)
        except Exception:
            unavailable = True
        if unavailable:
            raise VaultUnavailableError(self._UNAVAILABLE)
