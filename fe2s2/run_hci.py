#!/usr/bin/env python

import pyscf
import numpy as np
import argparse
import pyci

parser = argparse.ArgumentParser()
parser.add_argument('--fcidump', type=str, required=True)
parser.add_argument('--nelec', type=int, default=None)
parser.add_argument('--hci_epsilon_start', type=float, default=1.0)
parser.add_argument('--hci_epsilon_end', type=float, default=1e-7)
parser.add_argument('--hci_epsilon_factor', type=float, default=0.7943282347242815)
args = parser.parse_args()

ham = pyci.hamiltonian(args.fcidump)
beta = args.nelec // 2
alpha = args.nelec - beta
nelec = (alpha,beta)


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
while eps > args.hci_epsilon_end:
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
    eps = eps * args.hci_epsilon_factor
