
#####################################################
#
# PyRAI2MD 2 module for NequIP-NAC interface
#
# Author Menghang Wang, Chuin Wei Tan
# Oct 31 2025
#
######################################################

import torch
import numpy as np
import warnings

from nequip.model.inference_models import load_compiled_model
from nequip.nn import graph_model
from nequip.data import AtomicDataDict
from nequip.data.dict import from_dict
from ase.data import atomic_numbers
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from nequip.ase.nequip_calculator import _create_neighbor_transform

# Import keys from nequip_nac models
from nequip_nac._keys import (
    NAC_KEY,
    ENERGY_0_KEY,
    ENERGY_1_KEY,
    FORCE_0_KEY,
    FORCE_1_KEY,
)

class NequIPNAC:

    def __init__(self, param):
        self.param = param
        self.model_path = param['model_path']
        self.gpu = self.param['gpu']
        self.model = None
        self.metadata = None
        self.transforms = []
        self.nnac = param['nnac']
        self.natom = param['natom']

        self.set_device()
        self.load_model()

    def set_device(self):
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() and self.gpu else "cpu")
        
    def load_model(self):
        """Load trained and compiled NequIP-NAC model using from_compiled_model pattern"""
        from nequip.scripts._compile_utils import PAIR_NEQUIP_INPUTS

        # Define custom outputs for NAC model
        NAC_OUTPUTS = [
            ENERGY_0_KEY,
            ENERGY_1_KEY,
            FORCE_0_KEY,
            FORCE_1_KEY,
            NAC_KEY,
        ]
        
        # Load compiled model with proper inputs/outputs specification
        self.model, self.metadata = load_compiled_model(
            self.model_path, 
            device=self.device,
            input_keys=PAIR_NEQUIP_INPUTS,
            output_keys=NAC_OUTPUTS,
        )
        
        # Extract r_max and type_names from metadata for transforms
        r_max = self.metadata[graph_model.R_MAX_KEY]
        type_names = self.metadata[graph_model.TYPE_NAMES_KEY]
        
        # Create neighbor list transform with per-edge-type cutoffs if available
        neighbor_transform = _create_neighbor_transform(self.metadata, r_max, type_names)

        # Use type_names as chemical_symbols (fallback behavior)
        # You can pass chemical_symbols in param
        chemical_symbols = self.param.get('chemical_symbols', None)
        if chemical_symbols is None:
            warnings.warn(
                "Use model type names as chemical symbols; to avoid this warning, please provide the full `chemical_symbols` used in config.yaml during training."
            )
            chemical_symbols = type_names
        
        # Set up transforms for data preprocessing
        self.transforms = [
            ChemicalSpeciesToAtomTypeMapper(chemical_symbols),
            neighbor_transform,
        ]
        

    def _xyz_to_nequip_data(self, xyz_molecule) -> AtomicDataDict.Type:
        """
        Convert PyRAI2MD single MD frame to NequIP AtomicDataDict
        
        Args:
            xyz_molecule: np.array of shape (natom, 4) with [symbol, x, y, z]
        
        Returns:
            data: AtomicDataDict for NequIP model
        """
        # Extract symbols and coordinates
        symbols = xyz_molecule[:, 0]
        positions = xyz_molecule[:, 1:4].astype(np.float64)
        
        # Convert symbols to atomic numbers
        atom_numbers = np.array([atomic_numbers[s] for s in symbols])
        
        # Create NequIP data dict
        data = {
            AtomicDataDict.POSITIONS_KEY: positions,
            AtomicDataDict.ATOMIC_NUMBERS_KEY: atom_numbers,
            # For non-periodic systems, use default values
            AtomicDataDict.CELL_KEY: np.zeros((3, 3)),
            AtomicDataDict.PBC_KEY: np.array([False, False, False]),
        }
        
        return from_dict(data)
        
    def predict(self, xyz_list):
        """
        Predict energies, energy gradients, and NACs
        
        Args:
            xyz_list: List of structures, where each structure is (natom, 4)
                     - For single molecule: [[natom, 4]]
                     - For batch: [xyz1, xyz2, ...] where each xyz is (natom, 4)
        
        Returns:
            mean_dict: Dict with 'energy', 'energy_gradient' and 'nac' predictions
            std_dict: Dict with uncertainties (zeros for now)
        """
        # Check if xyz_list is a list
        if not isinstance(xyz_list, list):
            raise TypeError(
                f"xyz_list must be a list, got {type(xyz_list).__name__}. "
                f"For single molecule, use: predict([xyz_array])"
            )
        assert all(len(xyz) == self.natom for xyz in xyz_list), "All structures must have the same number of atoms as specified in natom."

        self.model.eval()

        num_data = len(xyz_list)

        # Prepare data
        data_list = [self._xyz_to_nequip_data(np.array(xyz)) for xyz in xyz_list]
        
        # Apply transforms (chemical species mapping + neighbor list) 
        for i in range(num_data):
            for t in self.transforms:
                data_list[i] = t(data_list[i])
            data_list[i] = AtomicDataDict.to_(data_list[i], self.device)

        # Use NequIP's built-in batching function
        data = AtomicDataDict.batched_from_list(data_list)
        
        # === predict + extract data ===
        out = self.model(data)

        # Extract for different states
        energy_0 = out[ENERGY_0_KEY].detach().cpu().numpy() 
        energy_1 = out[ENERGY_1_KEY].detach().cpu().numpy()
        energy_grad_0_all = -out[FORCE_0_KEY].detach().cpu().numpy()
        energy_grad_1_all = -out[FORCE_1_KEY].detach().cpu().numpy()
        # Extract NACs
        assert self.nnac == 1, "Only nnac=1 is supported currently."
        nac_all = out[NAC_KEY].detach().cpu().numpy()

        if num_data == 1:
            batch_idx = np.zeros(len(nac_all), dtype=int)
        else:
            batch_idx = out[AtomicDataDict.BATCH_KEY].cpu().numpy()

        # Unbatch node-level properties (SAFE for variable natom)
        energy_grad_list = []
        nacs_list = []
        
        for i in range(num_data):
            # Extract atoms belonging to structure i
            mask = batch_idx == i
            
            # Works regardless of natom for each structure
            energy_grad_0 = energy_grad_0_all[mask]  # (natom_i, 3)
            energy_grad_1 = energy_grad_1_all[mask]  # (natom_i, 3)
            nac = nac_all[mask]                     # (natom_i, 3)
            
            energy_grad_list.append(np.stack([energy_grad_0, energy_grad_1], axis=0))
            nacs_list.append(nac[np.newaxis])
        
        # Stack results
        energies = np.concatenate([energy_0, energy_1], axis=1) # (num_data, nstate)
        energy_grads = np.array(energy_grad_list)  # (num_data, nstate, natom, 3)
        nacs = np.array(nacs_list)  # (num_data, nnac, natom, 3)

        assert energies.shape == (num_data, 2)
        assert energy_grads.shape[0] == num_data and energy_grads.shape[1] == 2
        assert nacs.shape[0] == num_data and nacs.shape[1] == self.nnac

        mean_dict = {
            'energy': energies, 
            'energy_gradient': energy_grads,
            'nac': nacs
        }
        
        std_dict = {
            'energy': np.zeros_like(energies),
            'energy_gradient': np.zeros_like(energy_grads),
            'nac': np.zeros_like(nacs)
        }
        
        return mean_dict, std_dict