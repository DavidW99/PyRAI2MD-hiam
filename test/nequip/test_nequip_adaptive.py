import json

import numpy as np
import pytest

from PyRAI2MD.Machine_Learning import adaptive_sampling as adaptive_sampling_module
from PyRAI2MD.Machine_Learning.adaptive_sampling import AdaptiveSampling
from PyRAI2MD.Machine_Learning.training_data import Data
from PyRAI2MD.Keywords.key_control import KeyControl
from PyRAI2MD.Keywords.key_md import KeyMD
from PyRAI2MD.Keywords.key_molecule import KeyMolecule


def _write_dummy_model(tmp_path, name):
    path = tmp_path / name
    path.write_text("compiled model placeholder")
    return path


def _make_keywords(tmp_path, model_paths=None, **control_overrides):
    if model_paths is None:
        model_paths = [
            _write_dummy_model(tmp_path, "model-a.nequip.pth"),
            _write_dummy_model(tmp_path, "model-b.nequip.pth"),
        ]

    model_csv = ",".join(str(path) for path in model_paths)
    train_data = control_overrides.pop("train_data", None)

    nequip_eg = {
        "model_type": "eg",
        "chemical_symbols": ["H"],
    }
    nequip_nac = {
        "model_type": "nac",
        "chemical_symbols": ["H"],
    }

    keywords = {
        "control": {
            "title": "nequip-adaptive",
            "jobtype": "adaptive",
            "load": 1,
            "transfer": 0,
            "remote_train": 0,
            **control_overrides,
        },
        "version": "test",
        "molecule": {
            "ci": 2,
            "coupling": [[1, 2]],
        },
        "nequip": {
            "modeldir": model_csv,
            "eg_unit": "au",
            "nac_unit": "au",
            "gpu": 0,
            "silent": 1,
            "train_data": train_data,
            "chemical_symbols": ["H"],
            "nequip_eg": nequip_eg,
            "nequip_nac": nequip_nac,
        },
        "nequip_eg": nequip_eg,
        "nequip_nac": nequip_nac,
    }

    return keywords


def _make_adaptive_sampling_keywords(tmp_path, train_data=None, model_paths=None):
    keywords = _make_keywords(tmp_path, model_paths=model_paths, train_data=train_data)
    control = KeyControl().default().copy()
    md = KeyMD().default().copy()
    molecule = KeyMolecule().default().copy()

    control.update({
        "title": "nequip-adaptive",
        "qm": ["nequip"],
        "abinit": ["stubqc"],
        "jobtype": "adaptive",
        "ml_ncpu": 1,
        "qc_ncpu": 1,
        "gl_seed": 1,
        "load": 1,
        "transfer": 0,
        "remote_train": 0,
    })
    md.update({
        "ninitcond": 1,
        "method": "wigner",
        "format": "xyz",
        "gl_seed": 1,
        "temp": 300,
        "step": 1,
        "record_step": 1,
    })
    molecule.update({
        "ci": [2],
        "spin": [0],
        "coupling": [[1, 2]],
    })

    keywords["control"] = control
    keywords["md"] = md
    keywords["molecule"] = molecule

    return keywords


def _make_batch(energy=(0.3, 0.4)):
    return [
        np.array([[["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.8]]], dtype=object),
        np.array([[]], dtype=float),
        np.array([[]], dtype=float),
        np.array([[]], dtype=float),
        np.array([list(energy)], dtype=float),
        np.zeros((1, 2, 2, 3)),
        np.zeros((1, 1, 2, 3)),
        np.zeros((1, 0)),
    ]


def test_adaptive_sampling_init_with_nequip_allows_no_train_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        adaptive_sampling_module,
        "sampling",
        lambda *_args, **_kwargs: [np.array([["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.8]], dtype=object)],
    )
    monkeypatch.setattr(adaptive_sampling_module, "Trajectory", lambda x, keywords=None: x)
    monkeypatch.setattr(adaptive_sampling_module.multiprocessing, "set_start_method", lambda *args, **kwargs: None)

    adaptive = AdaptiveSampling(keywords=_make_adaptive_sampling_keywords(tmp_path, train_data=None))

    assert len(adaptive.initcond) == 1
    assert adaptive.data.natom == 0
    assert adaptive.data.nstate == 0


def test_update_train_set_noop_on_empty_qc_batch(monkeypatch):
    adaptive = AdaptiveSampling.__new__(AdaptiveSampling)
    adaptive.ml = "nequip"
    adaptive.itr = 0
    adaptive.data = Data()

    def fail_stat():
        raise AssertionError("stat should not run for empty QC batch")

    def fail_save(*_args, **_kwargs):
        raise AssertionError("save should not run for empty QC batch")

    monkeypatch.setattr(adaptive.data, "stat", fail_stat)
    monkeypatch.setattr(adaptive.data, "save", fail_save)

    empty_newdata = [
        np.array([], dtype=object),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
    ]

    ret = adaptive._update_train_set(empty_newdata)
    assert ret is adaptive
    assert len(adaptive.data.xyz) == 0


def test_update_train_set_uses_initialize_from_batch_for_first_nequip_batch(monkeypatch):
    adaptive = AdaptiveSampling.__new__(AdaptiveSampling)
    adaptive.ml = "nequip"
    adaptive.itr = 0
    adaptive.data = Data()

    stat_calls = {"count": 0}
    save_calls = {"count": 0}

    def stat_noop():
        stat_calls["count"] += 1
        return adaptive.data

    def save_noop(*_args, **_kwargs):
        save_calls["count"] += 1
        return adaptive.data

    def fail_append(_newdata):
        raise AssertionError("append should not run for the first NequIP QC batch")

    monkeypatch.setattr(adaptive.data, "stat", stat_noop)
    monkeypatch.setattr(adaptive.data, "save", save_noop)
    monkeypatch.setattr(adaptive.data, "append", fail_append)

    adaptive._update_train_set(_make_batch())

    assert adaptive.data.natom == 2
    assert adaptive.data.nstate == 2
    assert adaptive.data.nnac == 1
    assert adaptive.data.nsoc == 0
    assert stat_calls["count"] == 1
    assert save_calls["count"] == 1


def test_update_train_set_uses_append_after_first_nequip_batch(monkeypatch):
    adaptive = AdaptiveSampling.__new__(AdaptiveSampling)
    adaptive.ml = "nequip"
    adaptive.itr = 1
    adaptive.data = Data().initialize_from_batch(_make_batch())

    append_calls = {"count": 0}
    stat_calls = {"count": 0}
    save_calls = {"count": 0}

    append_impl = adaptive.data.append

    def append_and_count(newdata):
        append_calls["count"] += 1
        return append_impl(newdata)

    def stat_noop():
        stat_calls["count"] += 1
        return adaptive.data

    def save_noop(*_args, **_kwargs):
        save_calls["count"] += 1
        return adaptive.data

    def fail_initialize(_newdata):
        raise AssertionError("initialize_from_batch should not run after first NequIP batch")

    monkeypatch.setattr(adaptive.data, "append", append_and_count)
    monkeypatch.setattr(adaptive.data, "stat", stat_noop)
    monkeypatch.setattr(adaptive.data, "save", save_noop)
    monkeypatch.setattr(adaptive.data, "initialize_from_batch", fail_initialize)

    adaptive._update_train_set(_make_batch(energy=(0.5, 0.6)))

    assert append_calls["count"] == 1
    assert stat_calls["count"] == 1
    assert save_calls["count"] == 1
    assert adaptive.data.xyz.shape[0] == 2
    assert adaptive.data.energy.shape[0] == 2


@pytest.mark.parametrize(
    ("control_key", "value", "message"),
    [
        ("load", 0, "requires `load 1`"),
        ("transfer", 1, "does not support transfer learning"),
        ("remote_train", 1, "does not support remote training"),
    ],
)
def test_nequip_model_rejects_invalid_adaptive_settings(tmp_path, monkeypatch, control_key, value, message):
    monkeypatch.setattr(
        adaptive_sampling_module,
        "sampling",
        lambda *_args, **_kwargs: [np.array([["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.8]], dtype=object)],
    )
    monkeypatch.setattr(adaptive_sampling_module, "Trajectory", lambda x, keywords=None: x)
    monkeypatch.setattr(adaptive_sampling_module.multiprocessing, "set_start_method", lambda *args, **kwargs: None)

    keywords = _make_adaptive_sampling_keywords(tmp_path, train_data=None)
    keywords["control"][control_key] = value

    with pytest.raises(SystemExit, match=message):
        AdaptiveSampling(keywords=keywords)


def test_adaptive_sampling_search_with_stubqc_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    adaptive = AdaptiveSampling.__new__(AdaptiveSampling)
    adaptive.title = "nequip-adaptive"
    adaptive.version = "test-version"
    adaptive.maxiter = 1
    adaptive.ml = "nequip"
    adaptive.abinit = ["stubqc"]
    adaptive.dynsample = 0
    adaptive.itr = 0
    adaptive.ntraj = 1
    adaptive.nrefine = [0]
    adaptive.qc_ncpu = 1
    adaptive.atoms = np.array([["H"], ["H"]], dtype=object)
    adaptive.keywords = {
        "control": {"title": adaptive.title},
    }

    adaptive.data = Data()

    qc_called = {"value": False}

    def fail_if_training_runs(_):
        raise AssertionError("NequIP adaptive sampling should not call training")

    monkeypatch.setattr(adaptive, "_train_wrapper", fail_if_training_runs)

    adaptive._run_aimd = lambda: ["md-history"]

    def fake_checkpoint():
        with open(f"{adaptive.title}.log", "a") as log:
            log.write("adaptive checkpoint\n")
        with open(f"{adaptive.title}.adaptive.json", "w") as out:
            json.dump({"1": {"select": 1}}, out)
        return 0

    adaptive._checkpoint = fake_checkpoint

    def fake_run_abinit():
        qc_called["value"] = True
        geom_id, xyz, charges, cell, pbc, energy, grad, nac, soc, completion = adaptive._abinit_wrapper(
            [0, adaptive.select_cond[0]]
        )
        assert geom_id == 0
        assert completion == 1
        return [
            np.array([xyz], dtype=object),
            np.array([charges], dtype=float),
            np.array([cell], dtype=float),
            np.array([pbc], dtype=float),
            np.array([energy], dtype=float),
            np.array([grad], dtype=float),
            np.array([nac], dtype=float),
            np.array([soc], dtype=float),
        ]

    adaptive._run_abinit = fake_run_abinit

    mol = type("FakeMol", (), {})()
    mol.coord = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    mol.qm2_charge = np.zeros((0, 4))
    mol.cell = np.zeros((0, 3))
    mol.pbc = np.zeros(0)
    mol.nstate = 2
    mol.nnac = 1
    mol.nsoc = 0

    def fake_screen_error(_):
        adaptive.ntraj = 1
        adaptive.nrefine = [0]
        adaptive.select_cond = [mol]
        return adaptive

    adaptive._screen_error = fake_screen_error

    adaptive.search()

    assert qc_called["value"] is True
    assert (tmp_path / f"{adaptive.title}.log").exists()
    assert (tmp_path / f"{adaptive.title}.adaptive.json").exists()
    assert any(path.name.startswith("New-data") and path.suffix == ".json" for path in tmp_path.iterdir())
