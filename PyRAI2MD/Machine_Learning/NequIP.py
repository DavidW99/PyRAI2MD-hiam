
#####################################################
#
# PyRAI2MD 2 module for NequIP-NAC interface
#
# Author Menghang Wang, Chuin Wei Tan
# Oct 31 2025
#
######################################################

import random
from contextlib import contextmanager

import torch
import numpy as np
import warnings

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nequip.data import AtomicDataDict

class NequIPNAC:

    def __init__(self, param):
        self.param = param
        model_path = param['model_path']
        if isinstance(model_path, str):
            self.model_paths = [model_path]
        else:
            self.model_paths = model_path

        self.gpu = self.param['gpu']
        self.models = []
        self.metadata = None
        self.transforms = []
        self.nnac = param['nnac']
        self.natom = param['natom']

        self.set_device()

    def set_device(self):
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() and self.gpu else "cpu")

    @staticmethod
    def _get_nequip_load_dependencies():
        # Deferred imports keep this wrapper importable in lightweight test environments.
        from nequip.model.inference_models import load_compiled_model
        from nequip.nn import graph_model
        from nequip.ase.nequip_calculator import _create_neighbor_transform
        from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper
        from nequip.scripts._compile_utils import PAIR_NEQUIP_INPUTS

        return (
            load_compiled_model,
            graph_model,
            _create_neighbor_transform,
            ChemicalSpeciesToAtomTypeMapper,
            PAIR_NEQUIP_INPUTS,
        )

    @staticmethod
    def _get_nequip_predict_dependencies():
        from ase.data import atomic_numbers
        from nequip.data import AtomicDataDict
        from nequip.data.dict import from_dict

        return AtomicDataDict, from_dict, atomic_numbers

    @staticmethod
    def _get_nequip_nac_keys():
        from nequip_nac._keys import (
            NAC_KEY,
            ENERGY_0_KEY,
            ENERGY_1_KEY,
            FORCE_0_KEY,
            FORCE_1_KEY,
        )

        return NAC_KEY, ENERGY_0_KEY, ENERGY_1_KEY, FORCE_0_KEY, FORCE_1_KEY

    @staticmethod
    def _capture_torch_rng_state():
        state = {}

        try:
            state["torch_cpu"] = torch.get_rng_state()
        except AttributeError:
            pass

        try:
            if torch.cuda.is_available():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except (AttributeError, RuntimeError):
            pass

        return state

    @staticmethod
    def _restore_torch_rng_state(state):
        if "torch_cpu" in state:
            try:
                torch.set_rng_state(state["torch_cpu"])
            except AttributeError:
                pass

        if "torch_cuda" in state:
            try:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
            except (AttributeError, RuntimeError):
                pass

    @staticmethod
    def _capture_rng_state():
        state = {
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        state.update(NequIPNAC._capture_torch_rng_state())

        return state

    @staticmethod
    def _restore_rng_state(state):
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        NequIPNAC._restore_torch_rng_state(state)

    @contextmanager
    def _preserve_rng_state(self):
        state = self._capture_rng_state()
        try:
            yield
        finally:
            self._restore_rng_state(state)
        
    def load_model(self):
        """Load trained and compiled NequIP-NAC model using from_compiled_model pattern"""
        (
            load_compiled_model,
            graph_model,
            create_neighbor_transform,
            ChemicalSpeciesToAtomTypeMapper,
            PAIR_NEQUIP_INPUTS,
        ) = self._get_nequip_load_dependencies()
        NAC_KEY, ENERGY_0_KEY, ENERGY_1_KEY, FORCE_0_KEY, FORCE_1_KEY = self._get_nequip_nac_keys()

        # Define custom outputs for NAC model
        NAC_OUTPUTS = [
            ENERGY_0_KEY,
            ENERGY_1_KEY,
            FORCE_0_KEY,
            FORCE_1_KEY,
            NAC_KEY,
        ]
        
        # Load compiled model with proper inputs/outputs specification
        self.models = []
        first_metadata = None

        with self._preserve_rng_state():
            for i, path in enumerate(self.model_paths):
                model, metadata = load_compiled_model(
                    path,
                    device=self.device,
                    input_keys=PAIR_NEQUIP_INPUTS,
                    output_keys=NAC_OUTPUTS,
                )
                self.models.append(model)

                if i == 0:
                    first_metadata = metadata
                else:
                    # Check consistency of metadata across ensemble models
                    if metadata[graph_model.R_MAX_KEY] != first_metadata[graph_model.R_MAX_KEY]:
                        raise ValueError(f"Model at {path} has different r_max ({metadata[graph_model.R_MAX_KEY]}) than the first model ({first_metadata[graph_model.R_MAX_KEY]}). Ensemble models must have consistent metadata.")
                    
                    if metadata[graph_model.TYPE_NAMES_KEY] != first_metadata[graph_model.TYPE_NAMES_KEY]:
                        raise ValueError(f"Model at {path} has different type_names ({metadata[graph_model.TYPE_NAMES_KEY]}) than the first model ({first_metadata[graph_model.TYPE_NAMES_KEY]}). Ensemble models must have consistent metadata.")
                    
                    if metadata.get(graph_model.PER_EDGE_TYPE_CUTOFF_KEY, None) is not None:
                        if metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY] != first_metadata.get(graph_model.PER_EDGE_TYPE_CUTOFF_KEY, None):
                            raise ValueError(f"Model at {path} has different per_edge_type_cutoff than the first model. Ensemble models must have consistent metadata.")
        
        # Extract r_max and type_names from metadata for transforms
        self.metadata = first_metadata
        r_max = self.metadata[graph_model.R_MAX_KEY]
        type_names = self.metadata[graph_model.TYPE_NAMES_KEY]
        
        # Create neighbor list transform with per-edge-type cutoffs if available
        neighbor_transform = create_neighbor_transform(self.metadata, r_max, type_names)

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
        AtomicDataDict, from_dict, atomic_numbers = self._get_nequip_predict_dependencies()

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
        AtomicDataDict, _, _ = self._get_nequip_predict_dependencies()
        NAC_KEY, ENERGY_0_KEY, ENERGY_1_KEY, FORCE_0_KEY, FORCE_1_KEY = self._get_nequip_nac_keys()

        # Check if xyz_list is a list
        if not isinstance(xyz_list, list):
            raise TypeError(
                f"xyz_list must be a list, got {type(xyz_list).__name__}. "
                f"For single molecule, use: predict([xyz_array])"
            )
        assert all(len(xyz) == self.natom for xyz in xyz_list), "All structures must have the same number of atoms as specified in natom."

        num_data = len(xyz_list)

        # Prepare data
        data_list = [self._xyz_to_nequip_data(np.array(xyz)) for xyz in xyz_list]
        
        # Apply transforms (chemical species mapping + neighbor list) 
        for i in range(num_data):
            for t in self.transforms:
                data_list[i] = t(data_list[i])
            data_list[i] = AtomicDataDict.to_(data_list[i], self.device)

        if num_data == 1:
            data = data_list[0]
        else:
            # Use NequIP's built-in batching function
            data = AtomicDataDict.batched_from_list(data_list)
        
        # Lists to store predictions from each model
        all_energies = []
        all_energy_grads = []
        all_nacs = []

        for model in self.models:
            model.eval()
            
            # === predict + extract data ===
            out = model(data)

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

            all_energies.append(energies)
            all_energy_grads.append(energy_grads)
            all_nacs.append(nacs)

        # Calculate mean and std
        mean_energies = np.mean(all_energies, axis=0)
        std_energies = np.std(all_energies, axis=0, ddof=1) if len(self.models) > 1 else np.zeros_like(mean_energies)
        
        mean_energy_grads = np.mean(all_energy_grads, axis=0)
        std_energy_grads = np.std(all_energy_grads, axis=0, ddof=1) if len(self.models) > 1 else np.zeros_like(mean_energy_grads)
        
        mean_nacs = np.mean(all_nacs, axis=0)
        std_nacs = np.std(all_nacs, axis=0, ddof=1) if len(self.models) > 1 else np.zeros_like(mean_nacs)

        assert mean_energies.shape == (num_data, 2)
        assert mean_energy_grads.shape[0] == num_data and mean_energy_grads.shape[1] == 2
        assert mean_nacs.shape[0] == num_data and mean_nacs.shape[1] == self.nnac

        mean_dict = {
            'energy': mean_energies, 
            'energy_gradient': mean_energy_grads,
            'nac': mean_nacs
        }
        
        std_dict = {
            'energy': std_energies,
            'energy_gradient': std_energy_grads,
            'nac': std_nacs
        }
        
        return mean_dict, std_dict