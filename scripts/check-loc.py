#!/usr/bin/env python3
"""Per-file and per-folder line-of-code budget checker.

Reads `.loc-budget.toml` at the repo root for thresholds and overrides.
- Soft limits warn (locally and in CI).
- Hard limits fail when env `LOC_HARD=1` (set in CI, off locally).
- Files with `// loc-allow: <reason>` (Rust) or `# loc-allow: <reason>`
  in their first 50 lines are exempt from per-file checks; the reason
  is printed.

Run from the repo root:
    python3 scripts/check-loc.py            # soft mode (advisory)
    LOC_HARD=1 python3 scripts/check-loc.py # hard mode (CI; fails on >hard)
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / ".loc-budget.toml"
HARD_MODE = os.environ.get("LOC_HARD", "0") == "1"
ALLOW_RE = re.compile(r"^\s*(?://|#)\s*loc-allow:\s*(.+)$", re.MULTILINE)
SCAN_EXTS = (".rs", ".py")
SKIP_DIRS = {
    ".git",
    "target",
    "node_modules",
    "site",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    ".tox",
    ".ruff_cache",
    ".mypy_cache",
    ".pyright",
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit(f"error: {CONFIG_PATH} not found")
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def relpath(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT)).replace(os.sep, "/")


def best_override(path_str: str, overrides: dict) -> dict:
    """Most-specific (longest) glob-matching override wins; defaults fall through."""
    best: dict = {}
    best_specificity = -1
    for pattern, ovr in overrides.items():
        if fnmatch.fnmatch(path_str, pattern) and len(pattern) > best_specificity:
            best = ovr
            best_specificity = len(pattern)
    return best


def folder_match_override(folder_str: str, overrides: dict) -> dict:
    """A folder matches an override if its path is a prefix of the override's glob base."""
    best: dict = {}
    best_specificity = -1
    for pattern, ovr in overrides.items():
        base = pattern.rstrip("/").removesuffix("/**")
        if (folder_str == base or folder_str.startswith(base + "/")) and len(
            pattern
        ) > best_specificity:
            best = ovr
            best_specificity = len(pattern)
    return best


def discover_folders() -> list[Path]:
    folders: list[Path] = []
    for parent_rel in ("crates", "python/vpm"):
        parent = REPO_ROOT / parent_rel
        if parent.is_dir():
            folders.extend(sorted(p for p in parent.iterdir() if p.is_dir()))
    return folders


def count_lines(path: Path) -> int | None:
    # Two separate except clauses so the script stays valid on Python 3.11+;
    # PEP 758's `except X, Y:` parens-free syntax is 3.14-only and ruff
    # format (target-version=py314) would mangle a tuple form here.
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except UnicodeDecodeError:
        return None
    except OSError:
        return None


def loc_allow_reason(path: Path) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            head = "".join(f.readline() for _ in range(50))
    except UnicodeDecodeError:
        return None
    except OSError:
        return None
    m = ALLOW_RE.search(head)
    return m.group(1).strip() if m else None


def walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fname in sorted(filenames):
            if fname.endswith(SCAN_EXTS):
                yield Path(dirpath) / fname


def collect_breaches(
    defaults: dict, overrides: dict, folders_to_check: list[str]
) -> tuple[list[tuple[str, int, int, int, str | None]], list[tuple[str, int, int, int]]]:
    """Walk the tree, return (file_breaches, folder_breaches).

    A file breach is a (path, count, soft, hard, allow_reason) tuple where
    `count > soft`; allow_reason is the loc-allow exemption text or None.
    A folder breach is a (folder, total, soft, hard) tuple where total > soft.
    """
    file_breaches: list[tuple[str, int, int, int, str | None]] = []
    folder_loc: dict[str, int] = dict.fromkeys(folders_to_check, 0)

    for path in walk_files(REPO_ROOT):
        path_str = relpath(path)
        n = count_lines(path)
        if n is None:
            continue

        ovr = best_override(path_str, overrides)
        soft = ovr.get("file_soft", defaults["file_soft"])
        hard = ovr.get("file_hard", defaults["file_hard"])
        if n > soft:
            file_breaches.append((path_str, n, soft, hard, loc_allow_reason(path)))

        for folder_str in folders_to_check:
            if path_str.startswith(folder_str + "/"):
                folder_loc[folder_str] += n

    folder_breaches: list[tuple[str, int, int, int]] = []
    for folder_str, total in sorted(folder_loc.items()):
        ovr = folder_match_override(folder_str, overrides)
        soft = ovr.get("folder_soft", defaults["folder_soft"])
        hard = ovr.get("folder_hard", defaults["folder_hard"])
        if total > soft:
            folder_breaches.append((folder_str, total, soft, hard))

    return file_breaches, folder_breaches


def report_file_breaches(
    file_breaches: list[tuple[str, int, int, int, str | None]],
) -> tuple[int, int]:
    n_warn = 0
    n_err = 0
    for path_str, n, soft, hard, allow in file_breaches:
        if allow:
            print(f"  ALLOW  {path_str}: {n} lines (soft={soft}, hard={hard}) — loc-allow: {allow}")
            continue
        is_hard = n > hard
        if is_hard:
            n_err += 1
        else:
            n_warn += 1
        kind = "ERROR" if is_hard else "WARN "
        print(f"  {kind}  {path_str}: {n} lines (soft={soft}, hard={hard})")
    return n_warn, n_err


def report_folder_breaches(folder_breaches: list[tuple[str, int, int, int]]) -> tuple[int, int]:
    n_warn = 0
    n_err = 0
    for folder_str, total, soft, hard in folder_breaches:
        is_hard = total > hard
        if is_hard:
            n_err += 1
        else:
            n_warn += 1
        kind = "ERROR" if is_hard else "WARN "
        print(f"  {kind}  {folder_str}/: {total} lines (soft={soft}, hard={hard})")
    return n_warn, n_err


def main() -> int:
    cfg = load_config()
    defaults = cfg["defaults"]
    overrides = cfg.get("overrides", {})
    folders_to_check = [relpath(f) for f in discover_folders()]

    file_breaches, folder_breaches = collect_breaches(defaults, overrides, folders_to_check)
    fw, fe = report_file_breaches(file_breaches)
    dw, de = report_folder_breaches(folder_breaches)
    n_warn, n_err = fw + dw, fe + de

    mode = "HARD (CI)" if HARD_MODE else "SOFT (advisory)"
    if n_warn == 0 and n_err == 0:
        print(f"LOC budget OK ({len(folders_to_check)} folders, mode={mode}).")
        return 0

    print(f"\nSummary: {n_warn} warning(s), {n_err} error(s). Mode: {mode}.")
    return 1 if (HARD_MODE and n_err > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
