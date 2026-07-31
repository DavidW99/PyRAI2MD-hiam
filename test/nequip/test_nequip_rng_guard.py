import numpy as np
import pytest

from PyRAI2MD.Machine_Learning.NequIP import NequIPNAC


class _FakeGraphModel:
    R_MAX_KEY = "r_max"
    TYPE_NAMES_KEY = "type_names"
    PER_EDGE_TYPE_CUTOFF_KEY = "per_edge_type_cutoff"


def test_nequip_load_model_preserves_numpy_rng(monkeypatch):
    fake_metadata = {
        _FakeGraphModel.R_MAX_KEY: 4.5,
        _FakeGraphModel.TYPE_NAMES_KEY: ["H", "C", "N"],
    }

    def fake_load_compiled_model(path, device, input_keys=None, output_keys=None):
        # emulate NequIP's real `load_compiled_model`, which internally calls
        # `set_global_state` -> `seed_everything` (which reseeds the RNG)
        np.random.seed(123)
        return object(), fake_metadata

    monkeypatch.setattr(
        NequIPNAC,
        "_get_nequip_load_dependencies",
        staticmethod(
            lambda: (
                fake_load_compiled_model,                    # load_compiled_model
                _FakeGraphModel,                             # graph_model
                lambda species_map, type_names: {},          # handle_chemical_species_map
                lambda metadata, r_max, type_names, species_map: [lambda data: data],  # basic_transforms
                ["fake_input"],                              # PAIR_NEQUIP_INPUTS
            )
        ),
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
        }
    )
    model.load_model()

    observed_next_value = np.random.uniform(0, 1)

    # If the guard works, the next NumPy draw matches the untouched control stream.
    assert observed_next_value == pytest.approx(expected_next_value)
