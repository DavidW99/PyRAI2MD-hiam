import numpy as np
import pytest

from PyRAI2MD.Machine_Learning.NequIP import NequIPNAC


class _FakeGraphModel:
    R_MAX_KEY = "r_max"
    TYPE_NAMES_KEY = "type_names"
    PER_EDGE_TYPE_CUTOFF_KEY = "per_edge_type_cutoff"


class _IdentitySpeciesMapper:
    def __init__(self, chemical_symbols):
        self.chemical_symbols = chemical_symbols

    def __call__(self, data):
        return data


def test_nequip_load_model_preserves_numpy_rng(monkeypatch):
    fake_metadata = {
        _FakeGraphModel.R_MAX_KEY: 4.5,
        _FakeGraphModel.TYPE_NAMES_KEY: ["H", "C", "N"],
    }

    def fake_load_compiled_model(path, device, input_keys=None, output_keys=None):
        np.random.seed(123)
        return object(), fake_metadata

    monkeypatch.setattr(
        NequIPNAC,
        "_get_nequip_load_dependencies",
        staticmethod(
            lambda: (
                fake_load_compiled_model,
                _FakeGraphModel,
                lambda metadata, r_max, type_names: (lambda data: data),
                _IdentitySpeciesMapper,
                ["fake_input"],
            )
        ),
    )
    monkeypatch.setattr(
        NequIPNAC,
        "_get_nequip_nac_keys",
        staticmethod(lambda: ("nac", "energy_0", "energy_1", "force_0", "force_1")),
    )

    seed = 2025
    control_rng = np.random.RandomState(seed)
    expected_next_value = control_rng.uniform(0, 1)

    np.random.seed(seed)
    model = NequIPNAC(
        param={
            "model_path": "dummy.nequip.pth",
            "gpu": False,
            "nnac": 1,
            "natom": 3,
            "chemical_symbols": ["H", "C", "N"],
        }
    )
    model.load_model()

    observed_next_value = np.random.uniform(0, 1)

    # If the guard works, the next NumPy draw matches the untouched control stream.
    assert observed_next_value == pytest.approx(expected_next_value)
