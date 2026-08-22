from __future__ import annotations

from pathlib import Path

import pytest

from scripts.create_author_diagnostic_split import (
    SPLIT_MODE,
    create_author_diagnostic_split,
    write_split,
)
from spine_baseline.file_lists import AUTHOR_DIAGNOSTIC_WARNING


def _one_series_files(count: int = 20) -> list[str]:
    return [f"patient_001_t2_space_s{index:03d}.npz" for index in range(count)]


def test_author_diagnostic_split_is_deterministic_and_90_10() -> None:
    files = list(reversed(_one_series_files()))

    first = create_author_diagnostic_split(files, seed=42)
    second = create_author_diagnostic_split(list(reversed(files)), seed=42)

    assert first == second
    assert len(first.train_files) == 18
    assert len(first.validation_files) == 2
    assert not (set(first.train_files) & set(first.validation_files))
    assert first.shared_series == ("patient_001_t2_space",)


def test_author_diagnostic_split_writes_warning_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("\n".join(_one_series_files()) + "\n", encoding="utf-8")
    output = tmp_path / "split"

    split = create_author_diagnostic_split(_one_series_files(), seed=7)
    write_split(output, source, split)

    config = (output / "split_config.tsv").read_text(encoding="utf-8")
    assert f"split_mode\t{SPLIT_MODE}" in config
    assert "final_generalization_evidence\tfalse" in config
    assert "train_file_list_sha256\t" in config
    assert "validation_file_list_sha256\t" in config
    assert AUTHOR_DIAGNOSTIC_WARNING in config
    assert len((output / "train_files.txt").read_text().splitlines()) == 18
    assert len((output / "validation_files.txt").read_text().splitlines()) == 2


def test_author_diagnostic_split_rejects_invalid_source_and_existing_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two"):
        create_author_diagnostic_split(["only_s000.npz"], seed=42)
    with pytest.raises(ValueError, match="basenames only"):
        create_author_diagnostic_split(["../a_s000.npz", "a_s001.npz"], seed=42)

    source = tmp_path / "source.txt"
    source.write_text("\n".join(_one_series_files()) + "\n", encoding="utf-8")
    output = tmp_path / "split"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        write_split(output, source, create_author_diagnostic_split(_one_series_files(), seed=42))
