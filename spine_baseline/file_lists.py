from pathlib import Path


def read_file_list(path: Path) -> list[str]:
    """Read a non-empty, duplicate-free slice list used as experiment evidence."""
    files = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not files:
        raise ValueError(f"File list is empty: {path}")
    if len(files) != len(set(files)):
        raise ValueError(f"File list contains duplicates: {path}")
    return files
