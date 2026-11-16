"""Small, dependency-free provenance primitives for the revised branch."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_pairs, parse_constant=reject)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def workspace_path(workspace, relative, *, namespace=None):
    """Portable relative paths only; also reject junction/symlink escapes."""
    require(isinstance(relative, str) and relative and "\\" not in relative
            and ":" not in relative, "Use a portable workspace-relative path")
    p = PurePosixPath(relative)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == relative,
            "Path must be normalized and remain inside the workspace")
    root = Path(workspace).resolve(strict=True)
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root), "Path escapes workspace")
    if namespace:
        require(p.parts[:len(PurePosixPath(namespace).parts)] == PurePosixPath(namespace).parts,
                f"Path must be in {namespace}")
        # Output/source namespaces may not be redirected even to another place
        # inside this workspace (e.g. registrations -> historical phase7/runs).
        for length in range(1, len(p.parts) + 1):
            component = root.joinpath(*p.parts[:length])
            require(component.resolve() == component,
                    "Namespace components cannot be symlinks or junction redirects")
        base = root / namespace
        require(base.resolve().is_relative_to(root) and resolved.is_relative_to(base.resolve()),
                "Path escapes its output namespace")
    return resolved
