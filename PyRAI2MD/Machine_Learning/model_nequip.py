#####################################################
#
# PyRAI2MD 2 module for NequIP-NAC interface
#
# Author Menghang Wang, Chuin Wei Tan
# Oct 31 2025
#
######################################################

import os
import time
import sys
import torch.cuda
import numpy as np

from PyRAI2MD.Utils.timing import what_is_time
from PyRAI2MD.Utils.timing import how_long

from PyRAI2MD.Machine_Learning.NequIP import NequIPNAC


class NequIPModel:
    """NequIP-NAC interface for PyRAI2MD

    Parameters:          Type:
        keywords         dict        keywords dict
        id               int         calculation index
        runtype          str         'qm_high' or 'qm_high_mid_low'

    Attribute:           Type:
        model            NequIPNAC   NequIP-NAC model instance
        natom            int         number of atoms
        nstate           int         number of states
        nnac             int         number of NAC pairs
        nsoc             int         number of SOC pairs

    Functions:           Returns:
        load             self        load trained NN for prediction
        evaluate         self        run prediction
    """

    def __init__(self, keywords=None, job_id=None, runtype='qm_high_mid_low'):
        
        # Basic settings
        self.runtype = runtype
        title = keywords['control']['title']
        variables = keywords['nequip'].copy()
        eg_unit = variables['eg_unit']
        nac_unit = variables['nac_unit']
        gpu = variables['gpu']
        
        self.jobtype = keywords['control']['jobtype']
        self.version = keywords['version']
        self.silent = variables['silent']
        self.keywords = keywords

        # TODO: update to code standard
        self.natom = 'auto'
        self.nstate = keywords['molecule']['ci']
        self.nnac = len(keywords['molecule']['coupling'])
        self.nsoc = None 
        
        # Assign folder name
        if job_id is None or job_id == 1:
            self.name = f"NequIP-{title}"
        else:
            self.name = f"NequIP-{title}-{job_id}"
        
        # Unit conversions (au to eV/Å)
        h_to_ev = 27.211396132
        h_bohr_to_ev_a = 27.211396132 / 0.529177249
        
        print(f"NequIP-NAC model are trained and predicted in eV and eV/Å units.")
        # NequIP-NAC model ouputs in eV and eV/Å
        # PyRAI2MD expects Hartree and Hartree/Bohr

        if eg_unit == 'si':
            self.f_e = 1
            self.f_g = 1
        else:
            self.f_e = h_to_ev
            self.f_g = h_bohr_to_ev_a
        
        if nac_unit == 'si':
            self.f_n = 1
        else:
            self.f_n = h_bohr_to_ev_a

        # Setup GPU
        ngpu = torch.cuda.device_count()
        gpu = variables['gpu']
        
        if ngpu > 0 and gpu > 0:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        if ngpu > 0:
            self.device_name = torch.cuda.current_device()
        else:
            self.device_name = 'cpu'
        
        model_path = self._resolve_model_paths(variables)

        # Setup model parameters
        param = {
            'model_path': model_path,
            'gpu': gpu > 0,
            'nnac': self.nnac,
        }
        
        # Initialize NequIP-NAC model
        self.model = NequIPNAC(param)
        print(self._heading())

    def _resolve_model_paths(self, variables):
        # `modeldir` is a comma-separated list of standalone compiled model files, one per
        # ensemble member (vs the NNsMD backends' single library-managed directory).
        raw_modeldir = str(variables.get('modeldir', ''))
        modeldir_pieces = raw_modeldir.split(',')
        modeldir_paths = [p.strip() for p in modeldir_pieces if p.strip()]

        if len(modeldir_paths) == 0:
            sys.exit(
                '\n  KeywordError\n'
                '  PyRAI2MD: nequip requires model paths from `&nequip modeldir`.\n'
                '  For multiple models, use comma-separated paths without spaces, e.g.\n'
                '  `modeldir model1.nequip.pt2,model2.nequip.pt2`'
            )

        # `modeldir` keeps only the first whitespace token, so `a.pt2, b.pt2` becomes
        # 'a.pt2,', which splits into ['a.pt2', ''] -- the empty piece flags the drop.
        if any(piece.strip() == '' for piece in modeldir_pieces):
            sys.exit(
                '\n  KeywordError\n'
                '  PyRAI2MD: `&nequip modeldir` = %r has an empty comma-separated entry.\n'
                '  Paths must be comma-separated WITHOUT spaces (a space after a comma drops\n'
                '  the rest). Use `modeldir model1.nequip.pt2,model2.nequip.pt2`' % raw_modeldir
            )

        for path in modeldir_paths:
            if not os.path.exists(path):
                sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for nequip model %s' % path)

        if self.jobtype == 'adaptive' and len(modeldir_paths) < 2:
            sys.exit(
                '\n  KeywordError\n'
                '  PyRAI2MD: adaptive sampling with nequip requires at least two compiled models.\n'
                '  Provide comma-separated paths without spaces in `&nequip modeldir`, e.g.\n'
                '  `modeldir model1.nequip.pt2,model2.nequip.pt2`'
            )

        return modeldir_paths if len(modeldir_paths) > 1 else modeldir_paths[0]
    
    def _heading(self):
        headline = """
%s
 *---------------------------------------------------*
 |                                                   |
 |             NequIP-NAC model only                 |
 |                                                   |
 *---------------------------------------------------*

 Number of atoms:  %s
 Number of state:  %s
 Number of NAC:    %s
 Number of SOC:    %s (not supported yet)

 Device found: %s
 Running device: %s
 
""" % (
            self.version,
            self.natom,
            self.nstate,
            self.nnac,
            self.nsoc,
            self.device,
            self.device_name,
        )
        return headline

    def _set_natom_from_traj(self, traj, qm_region=False):
        if qm_region:
            natom = len(traj.qm_atoms)
        else:
            natom = len(traj.atoms)

        if self.natom == 'auto':
            # NequIP does not receive training data here, so cache natom from the first runtime structure.
            self.natom = natom
        elif self.natom != natom:
            sys.exit(
                '\n  ValueError\n  PyRAI2MD: NequIP received %s atoms, but the model was initialized with %s atoms'
                % (natom, self.natom)
            )

        return self.natom
    
    def load(self):
        """Load trained NequIP-NAC model"""
        self.model.load_model()
        
        return self

    def train(self):
        # Retraining is handled outside PyRAI2MD for this NequIP interface.
        sys.exit('\n  RuntimeError\n  PyRAI2MD: NequIP training/retraining is not supported in this interface')
    
    def _high(self, traj):
        """Run NequIP-NAC for high level (QM) region in QM/MM calculation"""
        traj = traj.apply_qmmm()

        natom = self._set_natom_from_traj(traj, qm_region=True)
        # Prepare input: (1, natom, 4) with [symbol, x, y, z]
        atoms = traj.qm_atoms.reshape((1, natom, 1))
        xyz = traj.qm_coord.reshape((1, natom, 3))
        x = np.concatenate((atoms, xyz), axis=-1).tolist()
        
        # Predict
        y_pred, y_std = self.model.predict(x)
        
        # Initialize return values
        energy = []
        gradient = []
        nac = []
        soc = []
        err_e = 0
        err_g = 0
        err_n = 0
        err_s = 0
        
        # Extract energy and gradient (NequIP-NAC predicts both states)
        # Note: [0] is used to extract the single batch result assumed during prediction
        if 'energy' in y_pred.keys():
            e_pred = np.array(y_pred['energy'])[0] / self.f_e  # (2,) -> two states
            e_std = np.array(y_std['energy'])[0] / self.f_e
            energy = e_pred
            err_e = np.amax(e_std)
        
        if 'energy_gradient' in y_pred.keys():
            g_pred = np.array(y_pred['energy_gradient'])[0] / self.f_g  # (2, natom, 3)
            g_std = np.array(y_std['energy_gradient'])[0] / self.f_g
            gradient = g_pred
            err_g = np.amax(g_std)
        
        # Extract NAC
        if 'nac' in y_pred.keys():
            n_pred = np.array(y_pred['nac'])[0] / self.f_n  # (1, natom, 3)
            n_std = np.array(y_std['nac'])[0] / self.f_n
            nac = n_pred
            err_n = np.amax(n_std)
        
        # SOC not supported
        if 'soc' in y_pred.keys():
            s_pred = np.array(y_pred['soc'])[0]
            s_std = np.array(y_std['soc'])[0]
            soc = s_pred
            err_s = np.amax(s_std)
        
        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s
    
    def _high_mid_low(self, traj):
        """Run NequIP-NAC for full system (all atoms) in pure QM calculation"""

        natom = self._set_natom_from_traj(traj, qm_region=False)
        # Prepare input: (1, natom, 4) with [symbol, x, y, z]
        atoms = traj.atoms.reshape((1, natom, 1))
        xyz = traj.coord.reshape((1, natom, 3))
        x = np.concatenate((atoms, xyz), axis=-1).tolist()
        
        # Predict
        y_pred, y_std = self.model.predict(x)
        
        # Initialize return values
        energy = []
        gradient = []
        nac = []
        soc = []
        err_e = 0
        err_g = 0
        err_n = 0
        err_s = 0
        
        # Extract energy and gradient
        # Note: [0] is used to extract the single batch result assumed during prediction
        if 'energy' in y_pred.keys():
            e_pred = np.array(y_pred['energy'])[0] / self.f_e  # (2,)
            e_std = np.array(y_std['energy'])[0] / self.f_e
            energy = e_pred
            err_e = np.amax(e_std)
        
        if 'energy_gradient' in y_pred.keys():
            g_pred = np.array(y_pred['energy_gradient'])[0] / self.f_g  # (2, natom, 3)
            g_std = np.array(y_std['energy_gradient'])[0] / self.f_g
            gradient = g_pred
            err_g = np.amax(g_std)
        
        # Extract NAC
        if 'nac' in y_pred.keys():
            n_pred = np.array(y_pred['nac'])[0] / self.f_n  # (1, natom, 3)
            n_std = np.array(y_std['nac'])[0] / self.f_n
            nac = n_pred
            err_n = np.amax(n_std)
        
        # SOC not supported
        if 'soc' in y_pred.keys():
            s_pred = np.array(y_pred['soc'])[0]
            s_std = np.array(y_std['soc'])[0]
            soc = s_pred
            err_s = np.amax(s_std)
        
        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s
    
    def evaluate(self, traj):
        """Main function to run NequIP-NAC and communicate with other PyRAI2MD modules"""
        
        if self.runtype == 'qm_high':
            energy, gradient, nac, soc, err_energy, err_grad, err_nac, err_soc = self._high(traj)
        else:
            energy, gradient, nac, soc, err_energy, err_grad, err_nac, err_soc = self._high_mid_low(traj)
        
        # Assign results to trajectory
        traj.energy = np.copy(energy)
        traj.grad = np.copy(gradient)
        traj.nac = np.copy(nac)
        traj.soc = np.copy(soc)
        traj.err_energy = err_energy
        traj.err_grad = err_grad
        traj.err_nac = err_nac
        traj.err_soc = err_soc
        traj.status = 1
        
        return traj
