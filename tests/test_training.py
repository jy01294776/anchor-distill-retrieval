from __future__ import annotations

from pathlib import Path

from anchor_distill.training import model_directory_sha256


def test_model_directory_hash_excludes_mutable_manifest(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text('{"run": 1}\n')
    first = model_directory_sha256(tmp_path)

    manifest.write_text('{"run": 2}\n')

    assert model_directory_sha256(tmp_path) == first
