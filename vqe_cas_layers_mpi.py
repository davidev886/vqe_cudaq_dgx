import numpy as np
import sys
import os
import pickle
import json
from src.vqe_cudaq_qnp_mpi import VqeQnp
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

    str_date_0 = datetime.today().strftime('%Y%m%d_%H%M%S')
    str_date =  options.get("data_dir", "")
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
    print("time for preparing the cudaq hamiltonian:", end-start)

    n_qubits = 2 * num_active_orbitals

    empty_orbitals = num_active_orbitals - ((num_active_electrons // 2) + (num_active_electrons % 2))
    # init_mo_occ = [2] * (num_active_electrons // 2) + [1] * (num_active_electrons % 2) + [0] * empty_orbitals

    n_alpha = int((num_active_electrons + spin) / 2)
    n_beta = int((num_active_electrons - spin) / 2)

    n_alpha_vec = np.array([1] * n_alpha + [0] * (num_active_orbitals - n_alpha))
    n_beta_vec = np.array([1] * n_beta + [0] * (num_active_orbitals - n_beta))
    init_mo_occ = (n_alpha_vec + n_beta_vec).tolist()

    system_name = f"FeNTA_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o_opt_{optimizer_type}"
    info_time = defaultdict(list)
    results = []
    for count_layer, n_vqe_layers in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]):

        print("# Start VQE with init_mo_occ", init_mo_occ, "layers", n_vqe_layers)
        time_start = time.time()
        vqe = VqeQnp(n_qubits=n_qubits,
                     n_layers=n_vqe_layers,
                     init_mo_occ=init_mo_occ,
                     target=target,
                     system_name=system_name)

        if count_layer == 0:
            if init_params:
                params = np.loadtxt(init_params)[1:]  # first row contains best energy
                options = {'maxiter': 50000,
                           'callback': True,
                           'optimizer_type': optimizer_type,
                           'initial_parameters': params}
            else:
                options = {'maxiter': 50000,
                           'callback': True,
                           'optimizer_type': optimizer_type}
        else:
            options = {'maxiter': 50000,
                       'callback': True,
                       'optimizer_type': optimizer_type,
                       'initial_parameters': params}

        energy_0, params, exp_vals = vqe.run_vqe_cudaq(hamiltonian_cudaq, options=options)
        energy = energy_0 + energy_core
        exp_vals = np.array(exp_vals) + energy_core
        exp_vals = np.reshape(exp_vals, (exp_vals.size, 1))
        print(energy, params)
        print()
        results.append([n_vqe_layers, energy])
        np.savetxt(f"{str_date}/energy_fenta_{basis.lower()}_"
                   f"cas_{num_active_electrons}e_"
                   f"{num_active_orbitals}o_"
                   f"opt_{optimizer_type}.dat",
                   np.array(results))

        np.savetxt(f"{str_date}/expvals_energy_fenta_{basis.lower()}_"
                   f"cas_{num_active_electrons}e_{num_active_orbitals}o_"
                   f"layer_{n_vqe_layers}_opt_{optimizer_type}.dat",
                   exp_vals)
        time_end = time.time()
        info_time["num_layer"].append(n_vqe_layers)
        info_time["time_vqe"].append(time_end - time_start)

        if len(info_time["num_layer"]) > 1:
            df = pd.DataFrame(info_time)
        else:
            df = pd.DataFrame(info_time, index=[0])

        df.to_csv(f'{str_date}/{system_name}_info_time_layers_opt_{optimizer_type}.csv', index=False)
        info_params = [energy] + np.array(params).tolist()
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
