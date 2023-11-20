import numpy as np
import sys
import os
import pickle
import json
from src.vqe_cudaq_qnp import VqeQnp
from src.utils_cudaq import get_cudaq_hamiltonian
import time

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
    hamiltonian_fname = options.get("hamiltonian_fname", 1)

    filehandler = open(hamiltonian_fname, 'rb')
    jw_hamiltonian = pickle.load(filehandler)
    start = time.time()
    hamiltonian_cudaq = get_cudaq_hamiltonian(jw_hamiltonian)
    end = time.time()
    print("time for preparing the cudaq hamiltonian:", end-start)

    n_qubits = 2 * num_active_orbitals

    empty_orbitals = num_active_orbitals - ((num_active_electrons // 2) + (num_active_electrons % 2))
    init_mo_occ = [2] * (num_active_electrons // 2) + [1] * (num_active_electrons % 2) + [0] * empty_orbitals

    results = []
    for n_vqe_layers in [1, 2 , 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        print("# Start VQE with init_mo_occ", init_mo_occ, "layers", n_vqe_layers)
        vqe = VqeQnp(n_qubits=n_qubits,
                     n_layers=n_vqe_layers,
                     init_mo_occ=init_mo_occ)

        energy, params, exp_vals = vqe.run_vqe_cudaq(hamiltonian_cudaq, options={'maxiter': 10000, 'callback': True})
        exp_vals = np.array(exp_vals)
        exp_vals = np.reshape(exp_vals, (exp_vals.size, 1))
        print(energy, params)
        print()
        results.append([n_vqe_layers, energy])
        np.savetxt(f"energy_fenta_{basis.lower()}_cas_{num_active_electrons}e_{num_active_orbitals}o.dat",
                   np.array(results))

        np.savetxt(f"expvals_energy_fenta_{basis.lower()}_"
                   f"cas_{num_active_electrons}e_{num_active_orbitals}o_"
                   f"layer_{n_vqe_layers}.dat",
                   exp_vals)


    np.savetxt(f"energy_fenta_{basis.lower()}_cas_{num_active_electrons}e_{num_active_orbitals}o.dat",
               np.array(results))
