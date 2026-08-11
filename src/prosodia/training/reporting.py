from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


def sanitize_json_value(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON null values."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    return value
