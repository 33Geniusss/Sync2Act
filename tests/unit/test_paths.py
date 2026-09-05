from datetime import datetime

import sync2act.paths as paths


def test_huggingface_cache_is_outside_dataset_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)

    cache = paths.huggingface_cache_root()

    assert cache == tmp_path / "Sync2Act" / ".cache" / "huggingface"
    assert cache.is_dir()


def test_default_dataset_path_uses_user_datasets_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)

    target = paths.default_dataset_path("lerobot/pusht")

    assert target == tmp_path / "Sync2Act" / "datasets" / "pusht"
    assert target.parent.is_dir()


def test_model_checkpoint_names_are_descriptive_and_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "installation_root", lambda: tmp_path)
    now = datetime(2026, 9, 4, 12, 34, 56)

    first = paths.new_model_checkpoint_path("bc_mlp", "action_noise", now=now)
    first.touch()
    second = paths.new_model_checkpoint_path("bc_mlp", "action_noise", now=now)

    assert first == tmp_path / "models" / "bc_mlp_action_noise_20260904_123456.pt"
    assert second == tmp_path / "models" / "bc_mlp_action_noise_20260904_123456_02.pt"
