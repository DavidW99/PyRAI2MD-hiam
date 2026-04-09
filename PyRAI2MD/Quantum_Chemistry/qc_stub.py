######################################################
#
# PyRAI2MD 2 module for stub QC interface
#
# Author OpenAI Codex
# Apr 8 2026
#
######################################################

import numpy as np


class StubQC:
    """Minimal QC stub used for adaptive-sampling smoke tests."""

    def __init__(self, keywords=None, job_id=None, runtype='qm_high_mid_low'):
        self.runtype = runtype
        self.job_id = job_id

    def appendix(self, _):
        return self

    def train(self):
        return self

    def load(self):
        return self

    def evaluate(self, traj):
        natom = len(traj.coord)
        nstate = traj.nstate
        nnac = traj.nnac
        nsoc = traj.nsoc

        coord = np.asarray(traj.coord, dtype=float)
        geom_scale = float(np.sum(coord * coord))

        energy = geom_scale * 1e-4 + np.arange(nstate, dtype=float) * 1e-3
        gradient = np.zeros((nstate, natom, 3))
        nac = np.zeros((nnac, natom, 3))
        soc = np.zeros(nsoc)

        traj.energy = np.copy(energy)
        traj.grad = np.copy(gradient)
        traj.nac = np.copy(nac)
        traj.soc = np.copy(soc)
        traj.err_energy = None
        traj.err_grad = None
        traj.err_nac = None
        traj.err_soc = None
        traj.status = 1

        return traj
