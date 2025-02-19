#!/usr/bin/env python

import pyscf
import numpy as np
import argparse
import pyci

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, required=True)
parser.add_argument('--hci_epsilon_start', type=float, default=1e-1)
parser.add_argument('--hci_epsilon_end', type=float, default=1e-7)
parser.add_argument('--hci_epsilon_factor', type=float, default=0.7943282347242815)
args = parser.parse_args()

base = args.base
ecore = np.load(f'{base}_ecore.npy')
h1 = np.load(f'{base}_h1.npy')
eri = np.load(f'{base}_eri.npy')
ham = pyci.hamiltonian(ecore, h1, eri.transpose(0,2,1,3))

molecule_base = '_'.join(base.split('_')[:-1])
with open('molecules.txt', 'r') as f:
    for line in f:
        molecule, _, _, _, _ = line.split()
        if molecule == molecule_base:
            molecule, electrons, ncas, charge, spin = line.split()
            electrons = int(electrons)
            ncas = int(ncas)
            charge = int(charge)
            spin = int(spin)
if spin == 0:
    nelec = (electrons//2, electrons//2)
elif spin == 1:
    nelec = (electrons//2+1, electrons//2)
elif spin == 2:
    nelec = (electrons//2+1, electrons//2-1)


wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
# hci wave function
wfn.add_hartreefock_det()
dets_added = 1
op = pyci.sparse_op(ham, wfn)
e_vals, e_vecs = op.solve(n=1, tol=1.0e-9)
old_energy = np.min(e_vals)
niter = 0

eps = args.hci_epsilon_start
print('Running HCI')
while eps >= args.hci_epsilon_end:
    dets_added = True
    while dets_added:
        # Add connected determinants to wave function via HCI
        dets_added = pyci.add_hci(ham, wfn, e_vecs[0], eps=eps)
        # Update CI matrix operator
        op.update(ham, wfn)
        # Solve CI matrix problem
        e_vals, e_vecs = op.solve(n=1)
        delta_e = old_energy - np.min(e_vals)
        old_energy = np.min(e_vals)
        niter += 1
        num_determinants = e_vecs.shape[1]
        print(f'{niter=} {eps=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')
    eps *= args.hci_epsilon_factor
