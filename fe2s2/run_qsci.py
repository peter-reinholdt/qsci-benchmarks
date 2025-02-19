
import pyscf
import pyscf.fci
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pyci
import hashlib
import os
import sh
import math

parser = argparse.ArgumentParser()
parser.add_argument('--fcidump', type=str, default=None, required=True)
parser.add_argument('--fcidump_nelec', type=int, default=None, required=True)
parser.add_argument('--repeats', type=int, default=10)
parser.add_argument('--civector', type=str, required=True)
args = parser.parse_args()

ham = pyci.hamiltonian(args.fcidump)
beta = args.fcidump_nelec //2
alpha = args.fcidump_nelec - beta
nelec = (alpha,beta)

wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
# hci wave function
op = pyci.sparse_op(ham, wfn)

# in pyscf format
p = np.load(args.civector).ravel()**2
indices = range(len(p))
dim = math.comb(ham.nbasis, beta)
def index_to_determinant(idx, dim=dim, norb=ham.nbasis, nelec=nelec):
    # get alpha/beta parts
    idx_alpha = idx // dim
    idx_beta = idx % dim
    alpha = pyscf.fci.cistring.addr2str(norb, nelec[0], idx_alpha)
    beta = pyscf.fci.cistring.addr2str(norb, nelec[1], idx_beta)
    return np.array([alpha, beta])

# sample determinants
sample_sizes = (10 ** np.arange(1,10, 0.1)).astype(np.int64)
fci_energy = -116.60560911938701
for N in sample_sizes:
    for _ in range(args.repeats):
        samples = np.random.choice(indices, p=p, size=N)
        unique = np.unique(samples)
        wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
        for index in unique:
            determinant = index_to_determinant(index)
            wfn.add_det(determinant)
            wfn.add_det(determinant[::-1])
        op = pyci.sparse_op(ham, wfn)
        sci_energies, sci_vectors = op.solve()
        num_dets = len(sci_vectors[0])
        determinant_fraction = num_dets / len(p)
        sci_energy = np.min(sci_energies) 
        energy_error = sci_energy - fci_energy
        print(f'Samples {N=} {len(unique)=} {num_dets=} {determinant_fraction=} {sci_energy=} {energy_error=}')

