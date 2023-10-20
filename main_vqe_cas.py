import numpy as np
import sys
import os
import pickle
import json
from src.vqe_cudaq_qnp import VqeQnp
from src.utils_cudaq import get_cudaq_hamiltonian

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
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

    filehandler = open(os.path.join(dir_path, hamiltonian_fname), 'rb')
    jw_hamiltonian = pickle.load(filehandler)

    hamiltonian_cudaq = get_cudaq_hamiltonian(jw_hamiltonian)

    n_qubits = 2 * num_active_orbitals

    empty_orbitals = num_active_orbitals - ((num_active_electrons // 2) + (num_active_electrons % 2))
    init_mo_occ = [2] * (num_active_electrons // 2) + [1] * (num_active_electrons % 2) + [0] * empty_orbitals

    print("Starting VQE")
    vqe = VqeQnp(n_qubits=n_qubits,
                 n_layers=10,
                 init_mo_occ=init_mo_occ)

    energy, params = vqe.run_vqe_cudaq(hamiltonian_cudaq, options={'maxiter': 10000, 'callback': True})

    print("Best energy, Best params")
    print(energy, params)
