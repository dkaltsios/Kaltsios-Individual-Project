from __future__ import annotations
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """
    Walk upwards from `start` until we find the repo root.
    Repo root is identified by having both:
      - README.md
      - Dataset/ directory
    """
    start = start or Path.cwd()

    for p in [start, *start.parents]:
        if (p / "README.md").exists() and (p / "Dataset").exists():
            return p

    raise FileNotFoundError(
        "Could not find project root. Expected to find README.md and Dataset/ in some parent directory."
    )


def dataset_dir(start: Path | None = None) -> Path:
    """Return the Dataset/ directory path."""
    return find_project_root(start) / "Dataset"


def notebooks_dir(start: Path | None = None) -> Path:
    """Return the notebooks/ directory path."""
    return find_project_root(start) / "notebooks"


def stage_dir(stage_name: str, start: Path | None = None) -> Path:
    """
    Return Stage directory path, e.g. stage_dir('Stage1') -> <root>/Stage1
    """
    root = find_project_root(start)
    p = root / stage_name
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist. Check stage_name.")
    return p
