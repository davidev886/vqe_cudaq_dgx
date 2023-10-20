import os
import sys
import numpy as np
import pickle
import json
from openfermion.transforms import jordan_wigner
from openfermion import generate_hamiltonian

def molecule_data(atom_name):
    # table B1 angstrom
    molecules = {'ozone': [('O', (0.0000000, 0.0000000, 0.0000000)),
                           ('O', (0.0000000, 0.0000000, 1.2717000)),
                           ('O', (1.1383850, 0.0000000, 1.8385340))]}

    return molecules[atom_name]


if __name__ == "__main__":

    with open(sys.argv[1]) as f:
        options = json.load(f)

    target = options.get("target", "nvidia")
    num_active_orbitals = options.get("num_active_orbitals", 5)
    num_active_electrons = options.get("num_active_electrons", 5)
    basis = options.get("basis", 'cc-pVTZ').lower()
    atom = options.get("atom", 'geo.xyz')
    dmrg = options.get("dmrg", 0)
    dmrg_states = options.get("dmrg_states", 1000)
    spin = options.get("spin", 1)

    hamiltonian_fname = f"ham_FeNTA_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o.pickle"
    print(hamiltonian_fname)

    if dmrg in (1, 'true'):
        dir_path = f"FeNTA_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o/dmrg_M_{dmrg_states}"
    else:
        dir_path = f"FeNTA_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o"

    h1 = np.load(os.path.join(dir_path, "h1.npy"))
    tbi = np.load(os.path.join(dir_path,"tbi.npy")) 
    energy_core = np.load(os.path.join(dir_path,"energy_core.npy"))
    
    mol_ham = generate_hamiltonian(h1, tbi, energy_core.item())
    jw_hamiltonian = jordan_wigner(mol_ham)
    filehandler = open(os.path.join(dir_path, hamiltonian_fname), 'wb')
    pickle.dump(jw_hamiltonian, filehandler)

