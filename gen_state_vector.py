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

    str_date_0 = datetime.today().strftime('%Y%m%d_%H%M%S')
    str_date =  options.get("data_dir", "")
    if len(str_date) == 0:
        str_date = str_date_0
    else:
        str_date = str_date + "_" + str_date_0

    os.makedirs(str_date, exist_ok=True)

    n_qubits = 2 * num_active_orbitals

    empty_orbitals = num_active_orbitals - ((num_active_electrons // 2) + (num_active_electrons % 2))
    # init_mo_occ = [2] * (num_active_electrons // 2) + [1] * (num_active_electrons % 2) + [0] * empty_orbitals

    n_alpha = int((num_active_electrons + spin) / 2)
    n_beta = int((num_active_electrons - spin) / 2)

    n_alpha_vec = np.array([1] * n_alpha + [0] * (num_active_orbitals - n_alpha))
    n_beta_vec = np.array([1] * n_beta + [0] * (num_active_orbitals - n_beta))
    init_mo_occ = (n_alpha_vec + n_beta_vec).tolist()
    params = np.loadtxt(init_params)[1: ]
    n_vqe_layers = len(params) // (n_qubits - 2)

    system_name = f"FeNTA_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o_opt_{optimizer_type}_wf"

    vqe = VqeQnp(n_qubits=n_qubits,
                 n_layers=n_vqe_layers,
                 init_mo_occ=init_mo_occ,
                 target=target,
                 system_name=system_name)

    state_vector = vqe.get_state_vector(params)
    np.savetxt(f"state_vec_{n_vqe_layers}.dat", state_vector)