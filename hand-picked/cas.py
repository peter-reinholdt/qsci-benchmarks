#!/usr/bin/env python

import pyscf
import pyscf.fci
import numpy as np
import sys
import os

xyz = sys.argv[1]
base = xyz[:-4]
with open(xyz, 'r') as f:
    f.readline()
    charge, mult = list(map(int, f.readline().split()))
    spin = mult - 1
basis = 'cc-pVDZ'

with open('molecules.txt', 'r') as f:
    for line in f:
        molecule, _, _, _, _ = line.split()
        if molecule == base:
            molecule, electrons, ncas, _charge, _spin = line.split()
            electrons = int(electrons)
            ncas = int(ncas)
            # read charge, spin from xyz 
#            charge = int(charge)
#            spin = int(spin)


m = pyscf.M(atom=xyz, basis=basis, spin=spin)
mf = pyscf.scf.RHF(m)
mf.verbose = 4
mf.kernel()
stab, _, is_stable, _ = mf.stability(return_status=True)
attempts = 0
while (not is_stable) or (not mf.converged):
    mf.kernel(stab)
    stab, _, is_stable, _ = mf.stability(return_status=True)
    attempts += 1
    if attempts > 50:
        break

pyscf.tools.fcidump.from_scf(mf, f'{base}_{basis}.fcidump')
cas = pyscf.mcscf.CASCI(mf, ncas=ncas, nelecas=electrons)
cas.fcisolver.max_space = 20
cas.fcisolver.max_cycle = 400
cas.fcisolver.nroots = 4

ncore = int((mf.mo_occ.sum() - electrons)//2)

h1, ecore = cas.h1e_for_cas()
eri = pyscf.ao2mo.full(m, mf.mo_coeff[:, ncore:ncore+ncas], aosym='1').reshape(ncas, ncas, ncas, ncas)
np.save(f'{base}_{basis}_ecore', ecore)
np.save(f'{base}_{basis}_h1', h1)
np.save(f'{base}_{basis}_eri', eri)

e_tot, e_cas, ci, mo_coeff, mo_energy = cas.kernel(verbose=5)
np.save(f'{base}_{basis}_civec', ci[0])

