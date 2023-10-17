import os
import sys
import numpy as np
import pickle
import json
from openfermion.transforms import jordan_wigner
from openfermion import generate_hamiltonian
from pyscf import gto, scf, ao2mo, mcscf, lib
from pyscf.lib import chkfile
# from pyscf.scf.chkfile import dump_scf

from src.vqe_cudaq_qnp import VqeQnp
from src.utils_cudaq import get_cudaq_hamiltonian

from pyscf import dmrgscf

def molecule_data(atom_name):
    # table B1 angstrom
    molecules = {'ozone': [('O', (0.0000000, 0.0000000, 0.0000000)),
                           ('O', (0.0000000, 0.0000000, 1.2717000)),
                           ('O', (1.1383850, 0.0000000, 1.8385340))]}

    return molecules[atom_name]


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    np.random.seed(12)
    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.BLOCKEXE_COMPRESS_NEVPT = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    with open(sys.argv[1]) as f:
        options = json.load(f)

    target = options.get("target", "nvidia")
    num_active_orbitals = options.get("num_active_orbitals", 5)
    num_active_electrons = options.get("num_active_electrons", 5)
    chkptfile = options.get("chkptfile", "mcscf.chk")
    basis = options.get("basis", 'cc-pVTZ').lower()
    atom = options.get("atom", 'geo.xyz')
    hamiltonian_fname = f"ham_FeNTA_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o.pickle"
    print(hamiltonian_fname)
    if 0:
        spin = 1
        multiplicity = spin + 1
        charge = 0

        mol = gto.M(
         atom=atom,
            spin=spin,
             charge=charge,
            basis=basis,
            verbose=4
            )

        mf = scf.ROHF(mol)
        mf.conv_tol = 1e-2
        mf.direct_scf_tol = 1e-3
        mf.kernel()

    geometry = molecule_data('ozone')
    basis = 'cc-pvqz'
    spin = 0
    multiplicity = spin + 1
    mol = gto.M(
        atom=geometry,
        basis=basis,
        spin=spin,
        verbose=4,
    )
    mf = scf.ROHF(mol)
    mf.kernel()
    chkptfile = None

    my_casci = mcscf.CASCI(mf, num_active_orbitals, num_active_electrons)
   # my_casci.fcisolver = dmrgscf.DMRGCI(mol)
    my_casci.fcisolver = dmrgscf.DMRGCI(mol, maxM=500, tol=1E-10)
    my_casci.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
    my_casci.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
    my_casci.fcisolver.threads = 8
    my_casci.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB
    my_casci.fcisolver.conv_tol = 1e-14

    x = (mol.spin/2 * (mol.spin/2 + 1))
    print(f"x={x}")
    my_casci.fix_spin_(ss=x)
    if chkptfile and os.path.exists(chkptfile):
        mo = chkfile.load(chkptfile, 'mcscf/mo_coeff')
        ecas, *_ = my_casci.kernel(mo)
    else:
        ecas, *_ = my_casci.kernel()

    print('FCI Energy in CAS:', ecas)

    h1, energy_core = my_casci.get_h1eff()
    h2 = my_casci.get_h2eff()
    h2_no_symmetry = ao2mo.restore('1', h2, num_active_orbitals)
    tbi = np.asarray(h2_no_symmetry.transpose(0, 2, 3, 1), order='C')

    mol_ham = generate_hamiltonian(h1, tbi, energy_core)
    jw_hamiltonian = jordan_wigner(mol_ham)
    # hamiltonian_cudaq = get_cudaq_hamiltonian(jw_hamiltonian)
    filehandler = open(hamiltonian_fname, 'wb')
    # hamiltonian_fname = f"ham_cudaq_O3_{num_active_electrons}e_{num_active_orbitals}o.pickle"
    pickle.dump(jw_hamiltonian, filehandler)
