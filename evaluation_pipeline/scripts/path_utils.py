"""Path handling shared by the portable evaluation scripts."""

from __future__ import annotations

from pathlib import Path


def resolve_path(value: str | Path, *, root: Path, base: Path | None = None) -> Path:
    """Resolve absolute, manifest-relative, and legacy repo-relative paths.

    New manifests should store paths relative to the manifest itself.  The
    repo-relative fallback keeps older ForgeWM manifests working.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if base is not None:
        candidates.append(base / path)
    candidates.append(root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()
