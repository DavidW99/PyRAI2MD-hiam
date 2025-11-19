
from ase.io import read, write
import numpy as np
from PyRAI2MD.Machine_Learning.NequIP import NequIPNAC

def test_nequip_data_conversion():
    """Test converstion of xyz_list to nequip data"""
    traj = read('molecule_eg.xyz', index=":")
    xyz_list = []
    for atoms in traj:
        natom = len(atoms)
        xyz_molecule = np.column_stack([atoms.get_chemical_symbols(), atoms.get_positions()])
        xyz_list.append(xyz_molecule)

    n_frames = len(xyz_list)
    nstate = 2  # assuming 2 states for this test
    nequip = NequIPNAC(param={
        'model_path': 'nac_model.nequip.pth',
        'gpu': False, # For testing, use CPU
        'nnac': 1,
        'natom': natom,
        'chemical_symbols': ['H', 'C', 'N']
    })
    nequip.load_model()

    mean_dict_batch, std_dict_batch = nequip.predict(xyz_list)
    mean_dict, std_dict = nequip.predict([xyz_list[0]])

    # Check dimensions
    assert mean_dict_batch['energy'].shape == (n_frames, nstate)
    assert mean_dict_batch['energy_gradient'].shape == (n_frames, nstate, natom, 3)
    assert mean_dict_batch['nac'].shape == (n_frames, 1, natom, 3)
    assert std_dict_batch['energy'].shape == (n_frames, nstate)
    assert std_dict_batch['energy_gradient'].shape == (n_frames, nstate, natom, 3)
    assert std_dict_batch['nac'].shape == (n_frames, 1, natom, 3)

    assert mean_dict['energy'].shape == (1, nstate)
    assert mean_dict['energy_gradient'].shape == (1, nstate, natom, 3)
    assert mean_dict['nac'].shape == (1, 1, natom, 3)
    assert std_dict['energy'].shape == (1, nstate)
    assert std_dict['energy_gradient'].shape == (1, nstate, natom, 3)
    assert std_dict['nac'].shape == (1, 1, natom, 3)

    # Check consistency between batched and single prediction
    np.testing.assert_allclose(mean_dict_batch['energy'][0], mean_dict['energy'][0], atol=1e-5)
    np.testing.assert_allclose(mean_dict_batch['energy_gradient'][0], mean_dict['energy_gradient'][0], atol=1e-5)
    np.testing.assert_allclose(mean_dict_batch['nac'][0], mean_dict['nac'][0], atol=1e-5)

if __name__ == "__main__":
    test_nequip_data_conversion()