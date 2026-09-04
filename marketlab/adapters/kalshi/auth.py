"""Kalshi production authentication: API key ID + RSA-PSS request signing.

Every authenticated request carries three headers:

``KALSHI-ACCESS-KEY``       the API key id (not secret - safe to log, but we don't).
``KALSHI-ACCESS-TIMESTAMP`` milliseconds since epoch, as a string.
``KALSHI-ACCESS-SIGNATURE`` base64(RSA-PSS-SHA256(timestamp_ms + METHOD + path)).

The signed message is ``f"{timestamp_ms}{method}{path}"`` where ``path`` includes the
``/trade-api/v2/...`` prefix and excludes the query string. Padding is PSS with MGF1(SHA256)
and salt length equal to the digest length (32 bytes) - Kalshi's documented scheme.

Nothing here ever logs the private key or the computed signature; ``marketlab.logging``
also redacts anything that looks like PEM material as a second line of defense.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from marketlab.logging import get_logger

log = get_logger(__name__)


class KalshiAuthError(Exception):
    """Raised for configuration/environment problems - never for a missing key file."""


class KalshiEnvironmentMismatchError(KalshiAuthError):
    """Raised when demo credentials would be sent to prod, or vice versa."""


def _looks_like_demo_host(base_url: str) -> bool:
    return "demo-api.kalshi" in base_url


class KalshiAuth:
    """Loads an RSA private key and signs Kalshi REST/WS requests.

    ``environment`` should be ``settings.secrets.kalshi_environment`` ("production" or
    "demo"). It is used purely as a safety interlock: :meth:`headers` (when given
    ``base_url``) refuses to sign a request whose target host doesn't match the
    configured environment, so a demo key can never be sent to production or vice versa.
    """

    def __init__(
        self,
        api_key_id: str,
        private_key_path: str,
        *,
        environment: str = "production",
        passphrase_env_var: str = "KALSHI_PRIVATE_KEY_PASSPHRASE",
    ) -> None:
        self._api_key_id = api_key_id
        self._private_key_path = private_key_path
        self.environment = environment.strip().lower() or "production"
        self._passphrase_env_var = passphrase_env_var
        self._private_key: rsa.RSAPrivateKey | None = None
        self._load_error: str | None = None

    @property
    def is_configured(self) -> bool:
        """False when the key id or key file is missing - never raises."""
        if not self._api_key_id or not self._private_key_path:
            return False
        return Path(self._private_key_path).is_file()

    def _ensure_key_loaded(self) -> rsa.RSAPrivateKey:
        if self._private_key is not None:
            return self._private_key
        key_path = Path(self._private_key_path)
        pem_bytes = key_path.read_bytes()
        passphrase = os.environ.get(self._passphrase_env_var, "") or None
        password = passphrase.encode("utf-8") if passphrase else None
        try:
            key = serialization.load_pem_private_key(pem_bytes, password=password)
        except TypeError:
            # cryptography raises TypeError when a password is required but not supplied,
            # or supplied but not required, depending on version - try the opposite.
            key = serialization.load_pem_private_key(
                pem_bytes, password=None if password else b""
            )
        if not isinstance(key, rsa.RSAPrivateKey):
            raise KalshiAuthError("Kalshi private key must be an RSA key")
        self._private_key = key
        return key

    def _check_environment(self, base_url: str) -> None:
        is_demo_host = _looks_like_demo_host(base_url)
        if self.environment == "demo" and not is_demo_host:
            raise KalshiEnvironmentMismatchError(
                "refusing to sign a production request with demo-configured credentials"
            )
        if self.environment != "demo" and is_demo_host:
            raise KalshiEnvironmentMismatchError(
                "refusing to sign a demo request with production-configured credentials"
            )

    def headers(self, method: str, path: str, base_url: str | None = None) -> dict[str, str]:
        """Build the three Kalshi auth headers for ``method``+``path``.

        ``path`` must include ``/trade-api/v2/...`` and must NOT include the query
        string. Pass ``base_url`` whenever it is known (the REST/WS adapters always know
        it) so the environment interlock in ``__init__`` can run.
        """
        if base_url is not None:
            self._check_environment(base_url)
        if not self.is_configured:
            raise KalshiAuthError("Kalshi auth is not configured (missing key id or key file)")

        key = self._ensure_key_loaded()
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}".encode()
        signature = key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256().digest_size),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }
