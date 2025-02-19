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
parser.add_argument('--base', type=str, required=True)
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--civector', type=str, required=True)
parser.add_argument('--electrons', type=int, required=True)
args = parser.parse_args()

base = args.base
ecore = np.load(f'{base}_ecore.npy')
h1 = np.load(f'{base}_h1.npy')
eri = np.load(f'{base}_eri.npy')
ham = pyci.hamiltonian(ecore, h1, eri.transpose(0,2,1,3))
nelec = (args.electrons//2, args.electrons//2)

wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
# hci wave function
op = pyci.sparse_op(ham, wfn)

# in pyscf format
p = np.abs(np.load(args.civector).ravel())**2
indices = range(len(p))
dim = math.comb(ham.nbasis, nelec[0])
def index_to_determinant(idx, dim=dim, norb=ham.nbasis, nelec=nelec):
    # get alpha/beta parts
    idx_alpha = idx // dim
    idx_beta = idx % dim
    alpha = pyscf.fci.cistring.addr2str(norb, nelec[0], idx_alpha)
    beta = pyscf.fci.cistring.addr2str(norb, nelec[1], idx_beta)
    return np.array([alpha, beta])

# sample determinants
#sample_sizes = (10 ** np.arange(1,10,1)).astype(np.int64)
sample_sizes = (10 ** np.arange(1,20,0.1)).astype(np.int64)
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
        print(f'Samples {N=} {len(unique)=} {num_dets=} {determinant_fraction=} {sci_energy=}')
