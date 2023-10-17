import os
import numpy as np
import pickle
import json
from openfermion.transforms import jordan_wigner
from openfermion import generate_hamiltonian
from pyscf import gto, scf, ao2mo, mcscf
from pyscf.lib import chkfile
# from pyscf.scf.chkfile import dump_scf

from src.vqe_cudaq_qnp import VqeQnp
from src.utils_cudaq import get_cudaq_hamiltonian


def molecule_data(atom_name):
    # table B1 angstrom
    molecules = {'ozone': [('O', (0.0000000, 0.0000000, 0.0000000)),
                           ('O', (0.0000000, 0.0000000, 1.2717000)),
                           ('O', (1.1383850, 0.0000000, 1.8385340))]}

    return molecules[atom_name]


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    writing_files = False
    np.random.seed(12)

    geometry = molecule_data('ozone')
    basis = 'cc-pvqz'
    spin = 0
    multiplicity = spin + 1
    num_active_orbitals = 9
    num_active_electrons = 12
    hamiltonian_fname = f"ham_cudaq_O3_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o.pickle"

    with open('input_vqe.json') as f:
        options = json.load(f)
    target = options.get("target", "nvidia")
    print(target)
    try:
        filehandler = open(hamiltonian_fname, 'rb')
        jw_hamiltonian = pickle.load(filehandler)
        hamiltonian_cudaq = get_cudaq_hamiltonian(jw_hamiltonian)
    except:
        mol = gto.M(
            atom=geometry,
            basis=basis,
            spin=spin,
            verbose=4,
        )
        hf = scf.ROHF(mol)
        hf.kernel()

        print("# HF Energy: ", hf.e_tot)
        my_casci = mcscf.CASCI(hf, num_active_orbitals, num_active_electrons)
        my_casci.kernel()

        ecas = my_casci.kernel()

        print('# FCI Energy in CAS:', ecas[0])

        h1, energy_core = my_casci.get_h1eff()
        h2 = my_casci.get_h2eff()
        h2_no_symmetry = ao2mo.restore('1', h2, num_active_orbitals)
        tbi = np.asarray(h2_no_symmetry.transpose(0, 2, 3, 1), order='C')
        print("# Start Hamiltonian generation")
        mol_ham = generate_hamiltonian(h1, tbi, energy_core)
        print("# Start JW transformation")
        jw_hamiltonian = jordan_wigner(mol_ham)
        print("# Start CUDAQ hamiltonian generation")
        hamiltonian_cudaq = get_cudaq_hamiltonian(jw_hamiltonian)

        filehandler = open(hamiltonian_fname, 'wb')
        # hamiltonian_fname = f"ham_cudaq_O3_{num_active_electrons}e_{num_active_orbitals}o.pickle"
        pickle.dump(jw_hamiltonian, filehandler)
        # mc = my_casci
        # casdm1, casdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
        # init_mo_occ = np.round(casdm1.diagonal())

    n_qubits = 2 * num_active_orbitals

    empty_orbitals = num_active_orbitals - ((num_active_electrons // 2) + (num_active_electrons % 2))
    init_mo_occ = [2] * (num_active_electrons // 2) + [1] * (num_active_electrons % 2) + [0] * empty_orbitals

    results = []
    for n_vqe_layers in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        print("# Start VQE with init_mo_occ", init_mo_occ, "layers", n_vqe_layers)
        vqe = VqeQnp(n_qubits=n_qubits,
                     n_layers=n_vqe_layers,
                     init_mo_occ=init_mo_occ,
                     target=target)
        print("# Start optimization VQE")
        energy, params = vqe.run_vqe_cudaq(hamiltonian_cudaq, options={'maxiter': 10000, 'callback': True})
        print(energy, params)
        print()
        results.append([n_vqe_layers, energy])
        np.savetxt(f"energy_ozone_{basis.lower()}_cas_{num_active_electrons}e_{num_active_orbitals}o.dat",
                   np.array(results))

    np.savetxt(f"energy_ozone_{basis.lower()}_cas_{num_active_electrons}e_{num_active_orbitals}o.dat",
               np.array(results))
