"""CPU timing utilities for Table 2."""

import time
from typing import Callable, Any, Tuple


def timed(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Run func(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed
