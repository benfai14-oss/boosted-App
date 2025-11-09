"""A lightweight caching layer for the global climate hedging project.

This module defines simple utilities to persist intermediate data
to disk, avoiding unnecessary network calls or expensive
computations.  It does not require any external services and
therefore works in constrained environments such as CI pipelines.
The cache is file‑based and stores pickled Python objects keyed by
human‑readable identifiers.

While more advanced caching systems (Redis, memcached) could be
used, the current scope of this project favours simplicity and
transparency.  The helper functions here are intentionally
stateless: they do not maintain global caches in memory, which
could lead to unexpected memory usage in long‑running
applications.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pickle

from scripts.common import write_audit_log

# Default cache directory.  Using the current working directory by
# default makes it easy to inspect cached files.  In production you
# may set this via an environment variable.
CACHE_DIR = Path(os.environ.get("CLIMATE_HEDGING_CACHE", ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_key(key: str) -> str:
    """Return a filesystem‑safe representation of the cache key.

    The raw key is hashed via SHA1 to avoid exposing sensitive
    information in file names and to prevent excessively long file
    names.  The first 10 characters of the hex digest are used to
    disambiguate keys with similar prefixes.

    Parameters
    ----------
    key : str
        An arbitrary string used to identify the cached object.

    Returns
    -------
    str
        Safe file name component.
    """
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"cache_{digest[:10]}"


def get_cache_path(key: str) -> Path:
    """Compute the path where a cache entry should be stored.

    Parameters
    ----------
    key : str
        Raw cache identifier.  For functions decorated by
        :func:`cached_call`, this will be generated from the
        function name and its arguments.

    Returns
    -------
    Path
        The full path to the cache file on disk.
    """
    return CACHE_DIR / (_safe_key(key) + ".pkl")


def cache_exists(key: str) -> bool:
    """Return ``True`` if a cached entry exists for the given key.
    """
    return get_cache_path(key).exists()


def load_from_cache(key: str) -> Any:
    """Load a previously cached object from disk.

    If the file does not exist or cannot be unpickled, ``None``
    is returned.  Any errors during deserialisation are
    swallowed; the caller should handle ``None`` to determine
    whether caching failed.
    """
    path = get_cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
            write_audit_log("cache", "load", {"key": key, "path": str(path)})
            return obj
    except Exception:
        return None


def save_to_cache(key: str, data: Any) -> None:
    """Serialise an object and write it to the cache directory.

    Overwrites any existing entry for the given key.  Uses
    ``pickle`` for serialisation, which is suitable for most
    Python objects but should not be used with untrusted data.
    """
    path = get_cache_path(key)
    try:
        with path.open("wb") as f:
            pickle.dump(data, f)
        write_audit_log("cache", "save", {"key": key, "path": str(path)})
    except Exception as exc:
        write_audit_log("cache", "error", {"key": key, "error": str(exc)})
        raise


def invalidate_cache(key: str) -> None:
    """Remove a cached entry if it exists.

    Deleting a cache entry frees up disk space and forces
    subsequent calls to recompute the value.  If the file does
    not exist, the function silently does nothing.
    """
    path = get_cache_path(key)
    if path.exists():
        try:
            path.unlink()
            write_audit_log("cache", "invalidate", {"key": key})
        except Exception:
            pass


def clear_cache() -> None:
    """Delete all cached entries in the cache directory.

    Use with caution: this removes *all* cached files without
    confirmation.
    """
    for f in CACHE_DIR.glob("cache_*.pkl"):
        try:
            f.unlink()
        except Exception:
            pass


def _make_call_key(func: Callable[[Any], Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
    """Generate a unique key based on a function and its arguments.

    The function name is combined with a JSON representation of
    positional and keyword arguments.  Because JSON requires
    serialisable types, non‑serialisable objects will be
    represented by their string representation.  The resulting
    string is hashed by :func:`_safe_key` when used for caching.
    """
    try:
        key_obj = {
            "func": func.__module__ + "." + func.__name__,
            "args": args,
            "kwargs": kwargs,
        }
        key_str = json.dumps(key_obj, default=str, sort_keys=True)
    except TypeError:
        key_str = str((func.__module__, func.__name__, args, kwargs))
    return key_str


def cached_call(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Decorator to transparently cache a function’s return value.

    When a cached function is called, a unique key is generated
    from the function’s name and its positional and keyword
    arguments.  If a cached result exists under that key, it is
    returned immediately.  Otherwise the function is executed and
    its result is stored on disk.  The cache key is independent of
    the order of keyword arguments.  Note that all positional
    arguments contribute to the key; changing a default argument
    will cause a cache miss.

    Because caching serialises objects with pickle, avoid using
    this decorator with functions that return open file handles or
    other non‑serialisable objects.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        raw_key = _make_call_key(func, args, kwargs)
        cache_key = _safe_key(raw_key)
        cached = load_from_cache(cache_key)
        if cached is not None:
            return cached
        result = func(*args, **kwargs)
        try:
            save_to_cache(cache_key, result)
        except Exception:
            # If caching fails, ignore and return the result
            pass
        return result

    # Expose the original function for introspection/testing
    wrapper.__wrapped__ = func  # type: ignore
    return wrapper
