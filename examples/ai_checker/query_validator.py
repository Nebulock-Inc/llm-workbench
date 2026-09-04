"""Query validation for ClickHouse SQL safety checks.

Copied from hunt-vault's ``nebulock_hunt_vault/core/query_validator.py`` (verbatim apart
from this note). It enforces cost-safety rules — time bounds, a LIMIT, no SQL comments,
early filtering — and is used as the first gate in ``ai_checker_pipeline.validate_sql``.
The optional ``query_sensors`` import is best-effort and simply no-ops here.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidationResult:
    """Result of query validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]
    estimated_row_count: Optional[int] = None


class QueryValidator:
    """Validates SQL queries for ClickHouse safety requirements.

    - Time bounds required (7 days max initially)
    - LIMIT clause required (start with 100)
    - No SQL comments
    - Early filtering recommended
    - Avoid multiple DISTINCT
    """

    TIME_PATTERNS = [
        r"toDate\(.+\)\s*>=\s*",
        r"toDateTime\(.+\)\s*>=\s*",
        r"timestamp\s*>=\s*",
        r"`?\btime\b`?\s*>=\s*",                       # time>= or `time` >= (backtick-quoted col)
        r"`?nebulock_ingest(?:ion)?_time`?\s*>=\s*",   # the ingestion-time bound
        r"WHERE.+>=\s*now\(\)\s*-\s*INTERVAL",
        r"WHERE.+>=\s*subtractDays",
        r"WHERE.+>=\s*subtractHours",
        r">=\s*toDateTime\(",
        r">=\s*parseDateTimeBestEffort",               # parseDateTimeBestEffort / ...64
    ]

    LIMIT_PATTERNS = [
        r"LIMIT\s+\d+",
        r"TOP\s+\d+",
    ]

    COMMENT_PATTERNS = [
        r"--",
        r"/\*.*?\*/",
    ]

    EXPENSIVE_PATTERNS = [
        (r"DISTINCT.*DISTINCT", "Multiple DISTINCT clauses detected (expensive operation)"),
        (r"SELECT\s+\*\s+FROM", "SELECT * detected (prefer explicit columns for performance)"),
        (r"GROUP BY.+ORDER BY.+LIMIT", "GROUP BY + ORDER BY without early filtering may be slow"),
    ]

    def __init__(self) -> None:
        """Initialize the query validator."""

    def validate(self, query: str, target: str = "clickhouse") -> ValidationResult:
        if target != "clickhouse":
            return ValidationResult(
                is_valid=False,
                errors=[f"Unsupported target: {target} (only 'clickhouse' is currently supported)"],
                warnings=[],
                info=[],
            )

        errors: List[str] = []
        warnings: List[str] = []
        info: List[str] = []

        normalized = " ".join(query.split())

        if self._has_comments(query):
            errors.append("SQL comments detected (-- or /* */). Remove them before execution.")

        if not self._has_time_bounds(normalized):
            errors.append(
                "No time bounds detected. Add time constraints (e.g., WHERE time >= now() - INTERVAL 7 DAY) "
                "to prevent timeouts."
            )
        else:
            info.append("Time bounds detected")

        is_count = self._is_count_only_query(normalized)
        limit_value = self._extract_limit(normalized)
        if limit_value is None and not is_count:
            errors.append("No LIMIT clause detected. Add LIMIT (start with 100, max 1000) to prevent large result sets.")
        elif limit_value is None and is_count:
            info.append("COUNT query detected - LIMIT not required")
        else:
            assert limit_value is not None
            if limit_value <= 100:
                info.append(f"LIMIT {limit_value} (safe - good starting point)")
            elif limit_value <= 1000:
                info.append(f"LIMIT {limit_value} (moderate)")
            else:
                warnings.append(f"LIMIT {limit_value} is high. Consider starting with LIMIT 100.")

        if not self._has_early_filtering(normalized):
            warnings.append(
                "No early filtering detected. Add WHERE clause before aggregations (GROUP BY, DISTINCT)."
            )
        else:
            info.append("Early filtering detected (WHERE clause present)")

        for pattern, message in self.EXPENSIVE_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                warnings.append(message)

        try:
            from nebulock_hunt_vault.core.query_sensors import QuerySensors  # type: ignore

            coverage_warnings = QuerySensors().check_coverage(query)
            for cw in coverage_warnings:
                warnings.append(f"Field sensor: {cw.message}")
        except Exception:
            pass  # Sensors are advisory / unavailable here; never block validation.

        if not re.search(r"FROM\s+[\w`.]+", normalized, re.IGNORECASE):
            errors.append("No FROM clause detected. Query must select from a table.")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings, info=info)

    def _has_comments(self, query: str) -> bool:
        if "--" in query:
            return True
        if re.search(r"/\*.*?\*/", query, re.DOTALL):
            return True
        return False

    def _has_time_bounds(self, normalized_query: str) -> bool:
        for pattern in self.TIME_PATTERNS:
            if re.search(pattern, normalized_query, re.IGNORECASE):
                return True
        return False

    def _is_count_only_query(self, normalized_query: str) -> bool:
        select_match = re.search(r"\bSELECT\s+(.*?)\s+FROM\b", normalized_query, re.IGNORECASE)
        if not select_match:
            return False
        select_clause = select_match.group(1).strip()
        expressions = [expr.strip() for expr in select_clause.split(",")]
        count_pattern = re.compile(r"^COUNT\s*\([^)]*\)(\s+(AS\s+)?\w+)?$", re.IGNORECASE)
        return all(count_pattern.match(expr) for expr in expressions)

    def _extract_limit(self, normalized_query: str) -> Optional[int]:
        limit_match = re.search(r"LIMIT\s+(\d+)", normalized_query, re.IGNORECASE)
        if limit_match:
            return int(limit_match.group(1))
        top_match = re.search(r"TOP\s+(\d+)", normalized_query, re.IGNORECASE)
        if top_match:
            return int(top_match.group(1))
        return None

    def _has_early_filtering(self, normalized_query: str) -> bool:
        where_match = re.search(r"WHERE", normalized_query, re.IGNORECASE)
        if not where_match:
            return False
        where_pos = where_match.start()
        group_match = re.search(r"GROUP\s+BY", normalized_query, re.IGNORECASE)
        if group_match and group_match.start() < where_pos:
            return False
        distinct_match = re.search(r"DISTINCT", normalized_query, re.IGNORECASE)
        if distinct_match and distinct_match.start() < where_pos:
            return False
        return True
