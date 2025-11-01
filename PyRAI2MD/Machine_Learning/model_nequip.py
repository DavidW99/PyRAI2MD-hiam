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
        modeldir = variables['modeldir']
        data = variables['data']
        eg_unit = variables['eg_unit']
        nac_unit = variables['nac_unit']
        
        self.jobtype = keywords['control']['jobtype']
        self.version = keywords['version']
        self.silent = variables['silent']
        self.natom = data.natom
        self.nstate = data.nstate
        self.nnac = data.nnac
        self.nsoc = data.nsoc
        
        # Assign folder name
        if job_id is None or job_id == 1:
            self.name = f"NequIP-{title}"
        else:
            self.name = f"NequIP-{title}-{job_id}"
        
        if modeldir is None or job_id not in [None, 1]:
            modeldir = self.name
        
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
        
        # Setup model parameters
        param = {
            'model_path': variables['model_path'],
            'gpu': gpu > 0,
            'natom': self.natom,
            'nnac': self.nnac,
            'chemical_symbols': variables.get('chemical_symbols', None),
        }
        
        # Initialize NequIP-NAC model
        self.model = NequIPNAC(param)
        print(self._heading())
    
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
    
    def load(self):
        """Load trained NequIP-NAC model"""
        self.model.load_model()
        
        return self
    
    def _high(self, traj):
        """Run NequIP-NAC for high level (QM) region in QM/MM calculation"""
        traj = traj.apply_qmmm()
        
        # Prepare input: (1, natom, 4) with [symbol, x, y, z]
        atoms = traj.qm_atoms.reshape((1, self.natom, 1))
        xyz = traj.qm_coord.reshape((1, self.natom, 3))
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
        if 'energy' in y_pred.keys():
            e_pred = np.array(y_pred['energy']) / self.f_e  # (2,) -> two states
            e_std = np.array(y_std['energy']) / self.f_e
            energy = e_pred
            err_e = np.amax(e_std)
        
        if 'energy_gradient' in y_pred.keys():
            g_pred = np.array(y_pred['energy_gradient']) / self.f_g  # (2, natom, 3)
            g_std = np.array(y_std['energy_gradient']) / self.f_g
            gradient = g_pred
            err_g = np.amax(g_std)
        
        # Extract NAC
        if 'nac' in y_pred.keys():
            n_pred = np.array(y_pred['nac']) / self.f_n  # (1, natom, 3)
            n_std = np.array(y_std['nac']) / self.f_n
            nac = n_pred
            err_n = np.amax(n_std)
        
        # SOC not supported
        if 'soc' in y_pred.keys():
            s_pred = np.array(y_pred['soc'])
            s_std = np.array(y_std['soc'])
            soc = s_pred
            err_s = np.amax(s_std)
        
        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s
    
    def _high_mid_low(self, traj):
        """Run NequIP-NAC for full system (all atoms) in pure QM calculation"""
        
        # Prepare input: (1, natom, 4) with [symbol, x, y, z]
        atoms = traj.atoms.reshape((1, self.natom, 1))
        xyz = traj.coord.reshape((1, self.natom, 3))
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
        if 'energy' in y_pred.keys():
            e_pred = np.array(y_pred['energy']) / self.f_e  # (2,)
            e_std = np.array(y_std['energy']) / self.f_e
            energy = e_pred
            err_e = np.amax(e_std)
        
        if 'energy_gradient' in y_pred.keys():
            g_pred = np.array(y_pred['energy_gradient']) / self.f_g  # (2, natom, 3)
            g_std = np.array(y_std['energy_gradient']) / self.f_g
            gradient = g_pred
            err_g = np.amax(g_std)
        
        # Extract NAC
        if 'nac' in y_pred.keys():
            n_pred = np.array(y_pred['nac']) / self.f_n  # (1, natom, 3)
            n_std = np.array(y_std['nac']) / self.f_n
            nac = n_pred
            err_n = np.amax(n_std)
        
        # SOC not supported
        if 'soc' in y_pred.keys():
            s_pred = np.array(y_pred['soc'])
            s_std = np.array(y_std['soc'])
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