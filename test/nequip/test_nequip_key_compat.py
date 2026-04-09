import pytest

from PyRAI2MD.Keywords.key_nequip import KeyNequIP


def test_nequip_legacy_keys_are_ignored_with_warning(capsys):
    reader = KeyNequIP(key_type="eg")

    result = reader.update(
        [
            "model_type eg",
            "chemical_symbols H C N",
            "natom 3",
            "model_path legacy.nequip.pth",
            "gpu 1",
        ]
    )

    assert result["model_type"] == "eg"
    assert result["chemical_symbols"] == ["H", "C", "N"]
    assert "natom" not in result
    assert "model_path" not in result
    assert "gpu" not in result

    stderr = capsys.readouterr().err
    assert "legacy key `natom`" in stderr
    assert "ignored" in stderr
    assert "auto-detected" in stderr
    assert "legacy key `model_path`" in stderr
    assert "legacy key `gpu`" in stderr
    assert "`&nequip modeldir` and `&nequip gpu`" in stderr


def test_nequip_unknown_key_still_fails():
    reader = KeyNequIP(key_type="nac")

    with pytest.raises(SystemExit, match="cannot recognize keyword unknown_key in &nequip_nac"):
        reader.update(["unknown_key 1"])
