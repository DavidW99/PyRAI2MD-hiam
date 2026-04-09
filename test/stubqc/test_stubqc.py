import numpy as np

from PyRAI2MD.Quantum_Chemistry.qc_stub import StubQC


def test_stubqc_evaluate_sets_expected_shapes():
    traj = type("FakeTraj", (), {})()
    traj.coord = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8], [1.0, 0.0, 0.0]]) # 3 atoms
    traj.nstate = 2
    traj.nnac = 1
    traj.nsoc = 0

    traj = StubQC().evaluate(traj)

    assert traj.energy.shape == (2,)
    assert traj.grad.shape == (2, 3, 3)
    assert traj.nac.shape == (1, 3, 3)
    assert traj.soc.shape == (0,)
    assert traj.status == 1
