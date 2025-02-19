import pyscf
import numpy as np
import ffsim
import math

# Function to obtain weights per determinants
def get_civec(state,norb,nelec):
    assert nelec[0] == nelec[1] # todo fix
    dim = math.comb(norb, nelec[0])
    civec = np.empty((dim,dim), dtype=np.complex128)
    bitmask= 2**norb - 1
    for nr, det in enumerate(ffsim.addresses_to_strings(range(ffsim.dim(norb,nelec)),norb,nelec,bitstring_type=ffsim.BitstringType.INT)):
        alpha = det & bitmask
        beta = (det >> norb) & bitmask
        alpha_idx = pyscf.fci.cistring.str2addr(norb, nelec[0], alpha)
        beta_idx = pyscf.fci.cistring.str2addr(norb, nelec[1], beta)
        civec[alpha_idx, beta_idx] = state[nr]
    return civec

#  Molecule, RHF, CCSD
basis = 'cc-pVDZ'
mol = pyscf.M(atom='N 0. 0. -0.545; N 0. 0. 0.545', basis=basis)
rhf = pyscf.scf.RHF(mol).run()
cc = pyscf.cc.CCSD(rhf, frozen=[0,1,-1,-2,-3,-4]).run(verbose=5)

print("Number of electrons: ", mol.nelec)
print("Number of orbitals:  ", mol.nao)

# Active space parameters
norb = 22
nelec = (5,5)

# 2-uCJ with CCSD-t2 amplitudes

# Construct UCJ operator
n_reps = 2 # layers
operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(cc.t2, n_reps=n_reps)

# Construct the Hartree-Fock state to use as the reference state
reference_state = ffsim.hartree_fock_state(norb, nelec)

# Apply the operator to the reference state
ansatz_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

print("saving...")
np.save('N2_ucj_statevec.npy', get_civec(ansatz_state,norb,nelec))
