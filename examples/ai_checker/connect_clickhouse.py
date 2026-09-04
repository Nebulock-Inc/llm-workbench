"""Self-contained ClickHouse connection for the ai-checker experiment.

Adapted from hunt-vault's ``nebulock_hunt_vault/core/clickhouse_connection.py``, trimmed
to be importable inside llm_lab with **no** ``nebulock_utils`` / ``config`` dependency.
Credentials come straight from environment variables (a ``.env`` at the llm_lab repo
root is auto-loaded by llm_lab), so local model-comparison runs need only:

    CLICKHOUSE_HOST=...            # e.g. abc123.us-east-1.aws.clickhouse.cloud
    CLICKHOUSE_PORT=8443           # optional (default 8443)
    CLICKHOUSE_USER=...            # a READ-ONLY account is strongly recommended
    CLICKHOUSE_PASSWORD=...        # (CLICKHOUSE_PWD also accepted)
    CLICKHOUSE_DATABASE=default    # optional
    CLICKHOUSE_SECURE=true         # optional

Tenant isolation is NOT handled here — it is applied per query by
``ai_checker_pipeline.inject_org_scope`` (server-side SQL rewrite). This connection
uses a single shared account; keep it read-only at the DB so a generated query can
never write, regardless of what the model emits.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional


class ClickHouseConfigError(Exception):
    """Configuration validation or loading error."""


class ClickHouseConnectionError(Exception):
    """Connection establishment or network error."""


class ClickHouseQueryError(Exception):
    """Query execution error."""


@dataclass
class ClickHouseConfig:
    host: str
    port: int
    username: str
    password: str
    database: str = "default"
    secure: bool = True

    @classmethod
    def load(cls) -> "ClickHouseConfig":
        """Load connection settings from environment variables."""
        host = os.getenv("CLICKHOUSE_HOST")
        if not host:
            raise ClickHouseConfigError(
                "CLICKHOUSE_HOST is not set. Add it (and CLICKHOUSE_USER/CLICKHOUSE_PASSWORD) "
                "to your .env or environment."
            )
        username = os.getenv("CLICKHOUSE_USER")
        password = os.getenv("CLICKHOUSE_PASSWORD") or os.getenv("CLICKHOUSE_PWD")
        if not username:
            raise ClickHouseConfigError("CLICKHOUSE_USER is not set.")
        if not password:
            raise ClickHouseConfigError("CLICKHOUSE_PASSWORD (or CLICKHOUSE_PWD) is not set.")

        port_raw = os.getenv("CLICKHOUSE_PORT", "8443") or "8443"
        try:
            port = int(port_raw)
        except (TypeError, ValueError) as exc:
            raise ClickHouseConfigError(f"CLICKHOUSE_PORT must be numeric, got {port_raw!r}.") from exc

        secure = str(os.getenv("CLICKHOUSE_SECURE", "true")).lower() in ("true", "1", "yes")
        database = os.getenv("CLICKHOUSE_DATABASE", "default") or "default"
        return cls(host=host, port=port, username=username, password=password, database=database, secure=secure)


class ClickHouseClient:
    """Thin wrapper over a lazily-created ``clickhouse_connect`` client.

    ``execute_query`` supports ClickHouse ``{name:Type}`` parameter binding and a
    per-query ``settings`` dict (used to carry the org-scope ``additional_table_filters``
    and safety limits injected by the pipeline).
    """

    def __init__(self, config: Optional[ClickHouseConfig] = None) -> None:
        self._config = config
        self._client: Any = None

    @property
    def config(self) -> ClickHouseConfig:
        if self._config is None:
            self._config = ClickHouseConfig.load()
        return self._config

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import clickhouse_connect
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ClickHouseConnectionError(
                "clickhouse-connect is not installed. Install the extra with: "
                "uv sync --extra clickhouse"
            ) from exc

        cfg = self.config
        is_lambda = os.environ.get("AWS_EXECUTION_ENV") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        try:
            self._client = clickhouse_connect.get_client(
                host=cfg.host,
                port=cfg.port,
                username=cfg.username,
                password=cfg.password,
                database=cfg.database,
                secure=cfg.secure,
                verify=not bool(is_lambda),
                connect_timeout=15,
                send_receive_timeout=120,
            )
            self._client.command("SELECT 1")
        except Exception as exc:
            if "auth" in str(exc).lower() or "credential" in str(exc).lower():
                raise ClickHouseConnectionError(
                    f"Authentication failed for user {cfg.username!r}; check CLICKHOUSE_USER/CLICKHOUSE_PASSWORD."
                ) from exc
            raise ClickHouseConnectionError(f"Failed to connect to ClickHouse at {cfg.host}: {exc}") from exc
        return self._client

    def execute_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run a SELECT and return ``{columns, data, rows, elapsed, query}``."""
        try:
            client = self._get_client()
            start = time.time()
            result = client.query(query, parameters=parameters or None, settings=settings or None)
            elapsed = time.time() - start
            return {
                "columns": list(result.column_names),
                "data": [list(row) for row in result.result_rows],
                "rows": len(result.result_rows),
                "elapsed": f"{elapsed:.3f}s",
                "query": query,
            }
        except Exception as exc:
            if "timeout" in str(exc).lower():
                raise ClickHouseQueryError(
                    f"Query timeout: {exc}\nTips: add time bounds (WHERE time >= now() - INTERVAL 7 DAY) "
                    "and a small LIMIT."
                ) from exc
            raise ClickHouseQueryError(f"Query execution failed: {exc}") from exc

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            finally:
                self._client = None


_CLIENT: Optional[ClickHouseClient] = None


def get_client() -> ClickHouseClient:
    """Return a process-wide singleton ClickHouseClient (lazy connection)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = ClickHouseClient()
    return _CLIENT
