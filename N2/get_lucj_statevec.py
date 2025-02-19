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


# Linear: same spin in connected line. other spin connected for same orbital
#pairs_aa = [(p, p + 1) for p in range(norb - 1)]
#pairs_ab = [(p, p) for p in range(norb)]
# Heavy-hex: same spin in connected zig-zag line. other spin connected for each 4th orbital
pairs_aa = [(p, p + 1) for p in range(norb - 1)]
pairs_ab = [(p, p) for p in range(0, norb, 4)]

interaction_pairs = (pairs_aa, pairs_ab)
# 2-uCJ with CCSD-t2 amplitudes

# Construct UCJ operator
n_reps = 2 # layers
operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(cc.t2, n_reps=n_reps, interaction_pairs=interaction_pairs)

# Construct the Hartree-Fock state to use as the reference state
reference_state = ffsim.hartree_fock_state(norb, nelec)

# Apply the operator to the reference state
ansatz_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

print("saving with t2-amplitudes...")
np.save('N2_lucj_statevec.npy', get_civec(ansatz_state,norb,nelec))
