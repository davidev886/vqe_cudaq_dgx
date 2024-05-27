"""
Contains the main file for running a complete VQE of the FeNTA system
"""
import numpy as np
import sys
import os
import pickle
import json
from src.vqe_cudaq_qnp import VqeQnp
from src.utils_cudaq import get_cudaq_hamiltonian
import time
from collections import defaultdict
import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)

    with open(sys.argv[1]) as f:
        options = json.load(f)

    target = options.get("target", "nvidia")
    print("target", target)
    num_active_orbitals = options.get("num_active_orbitals", 5)
    num_active_electrons = options.get("num_active_electrons", 5)
    basis = options.get("basis", 'cc-pVTZ').lower()
    atom = options.get("atom", 'geo.xyz')
    dmrg = options.get("dmrg", 0)
    dmrg_states = options.get("dmrg_states", 1000)
    spin = options.get("spin", 1)
    hamiltonian_fname = options.get("hamiltonian_fname", 1)
    optimizer_type = options.get("optimizer_type", "cudaq")
    start_layer = options.get("start_layer", 1)
    end_layer = options.get("end_layer", 10)
    init_params = options.get("init_params", None)
    mpi_support = options.get("mpi_support", False)

    str_date_0 = datetime.today().strftime('%Y%m%d_%H%M%S')
    str_date = options.get("data_dir", "")
    if len(str_date) == 0:
        str_date = str_date_0
    else:
        str_date = str_date + "_" + str_date_0

    os.makedirs(str_date, exist_ok=True)

    with open(f'{str_date}/{sys.argv[1]}', 'w') as f:
        json.dump(options, f, ensure_ascii=False, indent=4)

    filehandler = open(hamiltonian_fname, 'rb')
    jw_hamiltonian = pickle.load(filehandler)
    start = time.time()
    hamiltonian_cudaq, energy_core = get_cudaq_hamiltonian(jw_hamiltonian)
    end = time.time()
    print("# Time for preparing the cudaq hamiltonian:", end-start)

    n_qubits = 2 * num_active_orbitals

    empty_orbitals = num_active_orbitals - ((num_active_electrons // 2) + (num_active_electrons % 2))

    n_alpha = int((num_active_electrons + spin) / 2)
    n_beta = int((num_active_electrons - spin) / 2)

    n_alpha_vec = np.array([1] * n_alpha + [0] * (num_active_orbitals - n_alpha))
    n_beta_vec = np.array([1] * n_beta + [0] * (num_active_orbitals - n_beta))
    init_mo_occ = (n_alpha_vec + n_beta_vec).tolist()

    system_name = f"FeNTA_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o_opt_{optimizer_type}"
    info_time = defaultdict(list)

    options = {'maxiter': 50000,
               'optimizer_type': optimizer_type,
               'energy_core': energy_core,
               'mpi_support': mpi_support}

    results = []
    for count_layer, n_vqe_layers in enumerate(range(start_layer, end_layer + 1)):
        best_parameters = None
        print("# init_mo_occ", init_mo_occ)
        print("# layers", n_vqe_layers)
        time_start = time.time()
        vqe = VqeQnp(n_qubits=n_qubits,
                     n_layers=n_vqe_layers,
                     init_mo_occ=init_mo_occ,
                     target=target,
                     system_name=system_name)

        if count_layer == 0:
            if init_params:
                initial_parameters = np.loadtxt(init_params)[1:]  # first row contains best energy
                options['initial_parameters'] = initial_parameters
            else:
                options['initial_parameters'] = np.random.rand(vqe.num_params)
        else:
            # use as starting parameters the best from previous VQE
            options['initial_parameters'] = best_parameters

        energy_optimized, best_parameters, callback_energies = vqe.run_vqe_cudaq(hamiltonian_cudaq,
                                                                                 options=options)
        print(energy_optimized, best_parameters)

        results.append([n_vqe_layers, energy_optimized])
        np.savetxt(f"{str_date}/energy_fenta_{basis.lower()}_"
                   f"cas_{num_active_electrons}e_"
                   f"{num_active_orbitals}o_"
                   f"opt_{optimizer_type}.dat",
                   np.array(results))

        np.savetxt(f"{str_date}/callback_energies_fenta_{basis.lower()}_"
                   f"cas_{num_active_electrons}e_{num_active_orbitals}o_"
                   f"layer_{n_vqe_layers}_opt_{optimizer_type}.dat",
                   callback_energies)
        time_end = time.time()

        info_time["num_layer"].append(n_vqe_layers)
        info_time["time_vqe"].append(time_end - time_start)

        if len(info_time["num_layer"]) > 1:
            df = pd.DataFrame(info_time)
        else:
            df = pd.DataFrame(info_time, index=[0])

        df.to_csv(f'{str_date}/{system_name}_info_time_layers_opt_{optimizer_type}.csv', index=False)
        info_params = [energy_optimized] + np.array(best_parameters).tolist()
        np.savetxt(f"{str_date}/best_params_fenta_{basis.lower()}_"
                   f"cas_{num_active_electrons}e_"
                   f"{num_active_orbitals}o_"
                   f"layer_{n_vqe_layers}_opt_{optimizer_type}.dat",
                   np.array(info_params))

    np.savetxt(f"{str_date}/energy_fenta_{basis.lower()}_"
               f"cas_{num_active_electrons}e_"
               f"{num_active_orbitals}o_"
               f"opt_{optimizer_type}.dat",
               np.array(results))

    if len(info_time["num_layer"]) > 1:
        df = pd.DataFrame(info_time)
    else:
        df = pd.DataFrame(info_time, index=[0])

    df.to_csv(f'{str_date}/{system_name}_info_time_layers_opt_{optimizer_type}.csv', index=False)
