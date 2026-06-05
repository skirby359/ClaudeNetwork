"""Multi-client Microsoft Graph credential management.

Principles
----------
1. **One app registration per client.** Each engagement uses its own Azure AD app
   registration *in the client's own tenant*. Credentials are never shared across
   clients and never hardcoded in source.
2. **Secrets live in gitignored per-client files**, one file per client, under
   ``secrets/``. They are NEVER committed and NEVER stored in engagement profiles
   (profiles hold only non-secret config like internal domains and key dates).
3. **Profiles reference a client by slug**, not by secret. To run an engagement you
   load the profile (non-secret config) and, separately, the matching credentials.

Per-client file format — ``secrets/<slug>.env``::

    MS_TENANT_ID=...
    MS_APP_ID=...
    MS_APP_SECRET=...

Resolution order for ``load_graph_config(slug)``:
    1. ``secrets/<slug>.env`` (preferred — explicit per-client file)
    2. process environment / ``.env.local`` (the "active" client fallback)

Production hardening (beyond local files): store secrets in the OS keyring
(Windows Credential Manager via the ``keyring`` package) or Azure Key Vault, or
switch to certificate-based auth so there is no shared secret at all. This module
centralizes credential loading so that swap is a one-function change.
"""

import os
from pathlib import Path

from .msgraph import GraphConfig


SECRETS_DIR = Path(__file__).resolve().parent.parent.parent / "secrets"

_REQUIRED = ("MS_TENANT_ID", "MS_APP_ID", "MS_APP_SECRET")


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file, ignoring comments and blank lines."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def list_clients() -> list[str]:
    """List configured client slugs (one per secrets/<slug>.env file)."""
    if not SECRETS_DIR.exists():
        return []
    return sorted(p.stem for p in SECRETS_DIR.glob("*.env"))


def load_graph_config(client: str | None = None) -> GraphConfig:
    """Load Graph credentials for a client slug, or from the environment.

    Args:
        client: Client slug matching secrets/<slug>.env. If None, falls back to
            process environment variables (set from .env.local by the app).

    Raises:
        FileNotFoundError: named client has no secrets file.
        ValueError: required keys are missing.
    """
    if client:
        path = SECRETS_DIR / f"{client}.env"
        if not path.exists():
            raise FileNotFoundError(
                f"No credentials for client '{client}' (expected {path})"
            )
        values = _parse_env_file(path)
    else:
        values = {k: os.environ.get(k, "") for k in _REQUIRED}

    missing = [k for k in _REQUIRED if not values.get(k)]
    if missing:
        where = f"secrets/{client}.env" if client else "environment / .env.local"
        raise ValueError(f"Missing {', '.join(missing)} in {where}")

    return GraphConfig(
        tenant_id=values["MS_TENANT_ID"],
        app_id=values["MS_APP_ID"],
        app_secret=values["MS_APP_SECRET"],
    )
