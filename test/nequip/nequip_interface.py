
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

    nequip = NequIPNAC(param={
        'model_path': 'nac_model.nequip.pth',
        'gpu': False,
        'nnac': 1,
        'natom': natom,
        'chemical_symbols': ['H', 'C', 'N']
    })
    nequip.load_model()

    mean_dict, std_dict = nequip.predict(xyz_list)
    mean_dict, std_dict = nequip.predict([xyz_list[0]])

if __name__ == "__main__":
    test_nequip_data_conversion()