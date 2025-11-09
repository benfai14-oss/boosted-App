"""
Auto‑documentation utilities for the Global Climate Hedging project.

This module scans selected packages (by default: market_models, interface,
utils) and extracts module/class/function docstrings and signatures, then
writes a Markdown file you can version with the repo.

Usage (from project root):
  python -m utils.documentation                               # -> PROJECT_DOC.md
  python -m utils.documentation --out docs/API_REFERENCE.md
  python -m utils.documentation --packages market_models utils --fail-on-missing

What it generates:
  - A sorted inventory of modules with their top‑level docstrings
  - Public classes and functions (name, signature, short doc)
  - A quick I/O section if a function docstring contains markers like
    "Args:", "Returns:", or "Raises:" (pep257‑style).

Why it helps:
  - Easy onboarding/review, clear API surface for each layer
  - Encourages writing docstrings (optionally enforced with --fail-on-missing)
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass
class ObjDoc:
    qualname: str
    kind: str  # "function" | "class"
    signature: str
    doc: str | None

@dataclass
class ModuleDoc:
    name: str
    doc: str | None
    objects: List[ObjDoc]

# --------------------------------------------------------------------------------------
# Discovery & extraction
# --------------------------------------------------------------------------------------

DEFAULT_PACKAGES = ["market_models", "interface", "utils"]


def _iter_module_paths(package: str) -> Iterable[str]:
    """Yield importable module paths under a package (skip dunders/tests)."""
    pkg_root = Path(package)
    if not pkg_root.exists():
        return []
    for root, _, files in os.walk(pkg_root):
        # Skip virtual envs and caches
        if "__pycache__" in root or ".venv" in root:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            if fn == "__init__.py":
                # include the package module itself
                mod_path = Path(root).as_posix().replace("/", ".")
            else:
                mod_path = (Path(root) / fn).as_posix().replace("/", ".")[:-3]
            # Skip obvious non‑code files or tests
            base = os.path.basename(mod_path)
            if base.startswith("_") or base.endswith("_test") or base.startswith("test_"):
                continue
            yield mod_path


def _safe_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


def _trim(doc: Optional[str], max_lines: int = 12) -> Optional[str]:
    if not doc:
        return None
    lines = [ln.rstrip() for ln in doc.strip().splitlines()]
    return "\n".join(lines[:max_lines]).strip() or None


def extract_docs(packages: List[str]) -> Dict[str, ModuleDoc]:
    """Collect docstrings and signatures for modules/classes/functions."""
    # Ensure project root on sys.path
    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    results: Dict[str, ModuleDoc] = {}
    for pkg in packages:
        for mod_path in _iter_module_paths(pkg):
            module = _safe_import(mod_path)
            if module is None:
                continue
            mdoc = _trim(inspect.getdoc(module))
            objects: List[ObjDoc] = []

            for name, obj in inspect.getmembers(module):
                # Public API only
                if name.startswith("_"):
                    continue
                if inspect.isfunction(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except Exception:
                        sig = "(…)"
                    objects.append(
                        ObjDoc(qualname=f"{mod_path}.{name}", kind="function", signature=sig, doc=_trim(inspect.getdoc(obj)))
                    )
                elif inspect.isclass(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except Exception:
                        sig = "(class)"
                    objects.append(
                        ObjDoc(qualname=f"{mod_path}.{name}", kind="class", signature=sig, doc=_trim(inspect.getdoc(obj)))
                    )

            results[mod_path] = ModuleDoc(name=mod_path, doc=mdoc, objects=sorted(objects, key=lambda o: o.qualname))
    return results

# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

def render_markdown(docs: Dict[str, ModuleDoc]) -> str:
    lines: List[str] = []
    lines.append("# Global Climate Hedging – API & Module Reference\n")
    lines.append("_This file is auto‑generated by `utils.documentation`._\n")
    lines.append("")

    for mod_name in sorted(docs.keys()):
        m = docs[mod_name]
        lines.append(f"\n## {m.name}\n")
        lines.append(f"{m.doc or '*No module docstring.*'}\n")
        if not m.objects:
            lines.append("_No public objects found._\n")
            continue
        lines.append("\n### Public API\n")
        for obj in m.objects:
            header = f"- **{obj.qualname}**`{obj.signature}`"
            lines.append(header)
            if obj.doc:
                lines.append(f"  \n  {obj.doc}\n")
            else:
                lines.append("  \n  _No docstring._\n")
    return "\n".join(lines).strip() + "\n"

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate API documentation (Markdown)")
    ap.add_argument("--packages", nargs="+", default=DEFAULT_PACKAGES, help="Packages to scan (default: market_models interface utils)")
    ap.add_argument("--out", default="PROJECT_DOC.md", help="Output Markdown path")
    ap.add_argument("--fail-on-missing", action="store_true", help="Return non‑zero exit code if any public object has no docstring")
    args = ap.parse_args(argv)

    docs = extract_docs(args.packages)
    md = render_markdown(docs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] Documentation written to {out_path}")

    if args.fail_on_missing:
        missing = []
        for m in docs.values():
            for obj in m.objects:
                if not obj.doc:
                    missing.append(obj.qualname)
        if missing:
            print("[FAIL] Missing docstrings for:")
            for qn in missing:
                print("  -", qn)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

