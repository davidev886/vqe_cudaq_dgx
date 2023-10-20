import os
import sys
import numpy as np
#import pickle
import json
# from openfermion.transforms import jordan_wigner
# from openfermion import generate_hamiltonian
from pyscf import gto, scf, ao2mo, mcscf, lib
from pyscf.lib import chkfile
# from pyscf.scf.chkfile import dump_scf

#from src.vqe_cudaq_qnp import VqeQnp
#from src.utils_cudaq import get_cudaq_hamiltonian

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
    #dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    #dmrgscf.settings.BLOCKEXE_COMPRESS_NEVPT = os.popen("which block2main").read().strip()
    #dmrgscf.settings.MPIPREFIX = ''

    with open(sys.argv[1]) as f:
        options = json.load(f)

    target = options.get("target", "nvidia")
    num_active_orbitals = options.get("num_active_orbitals", 5)
    num_active_electrons = options.get("num_active_electrons", 5)
    chkptfile_rohf = options.get("chkptfile_rohf", None)
    chkptfile_cas = options.get("chkptfile_cas", None)
    basis = options.get("basis", 'cc-pVTZ').lower()
    atom = options.get("atom", 'geo.xyz')
    dmrg = options.get("dmrg", 0)
    dmrg_states = options.get("dmrg_states", 1000)
    spin = options.get("spin", 1)
    hamiltonian_fname = f"ham_FeNTA_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o.pickle"
    
    print(hamiltonian_fname)
    
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
    if chkptfile_rohf and os.path.exists(chkptfile_rohf):
        dm = mf.from_chk(chkptfile_rohf)
        # mf.max_cycle = 0
        mf.kernel(dm)
    else:
        mf.kernel()

    my_casci = mcscf.CASCI(mf, num_active_orbitals, num_active_electrons)
    if dmrg in (1, 'true'):
        dir_path = f"FeNTA_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o/dmrg_M_{dmrg_states}"
        my_casci.fcisolver = dmrgscf.DMRGCI(mol, maxM=dmrg_states, tol=1E-10)
        my_casci.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
        my_casci.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
        # my_casci.fcisolver.threads = 8
        my_casci.fcisolver.memory = int(mol.max_memory / 1000)  # mem in GB
        my_casci.fcisolver.conv_tol = 1e-14
    else:
        dir_path = f"FeNTA_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o"
        x = (mol.spin / 2 * (mol.spin / 2 + 1))
        print(f"x={x}")
        my_casci.fix_spin_(ss=x)

    os.makedirs(dir_path, exist_ok=True)

    if chkptfile_cas and os.path.exists(chkptfile_cas):
        mo = chkfile.load(chkptfile_cas, 'mcscf/mo_coeff')
        ecas, *_ = my_casci.kernel(mo)
    else:
        ecas, *_ = my_casci.kernel()

    print('FCI Energy in CAS:', ecas)

    h1, energy_core = my_casci.get_h1eff()
    h2 = my_casci.get_h2eff()
    h2_no_symmetry = ao2mo.restore('1', h2, num_active_orbitals)
    tbi = np.asarray(h2_no_symmetry.transpose(0, 2, 3, 1), order='C')

    # os.makedirs(dir_path, exist_ok=True)
    print(os.path.join(dir_path, "h1.npy"))
    np.save(os.path.join(dir_path, "h1.npy"), h1)
    np.save(os.path.join(dir_path, "tbi.npy"), tbi)
    np.save(os.path.join(dir_path, "energy_core.npy"), energy_core)

