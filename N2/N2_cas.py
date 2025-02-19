#!/usr/bin/env python

import pyscf
import numpy as np
import sys

xyz = sys.argv[1]
base = xyz[:-4]
basis = 'cc-pVDZ'

ncore = 2
ncas = 22

m = pyscf.M(atom=xyz, basis=basis)
m.max_memory = 300000 # 300 GB
mf = pyscf.scf.RHF(m).run()
pyscf.tools.fcidump.from_scf(mf, f'{base}_{basis}.fcidump')
cas = pyscf.mcscf.CASCI(mf, ncas, 10)

h1, ecore = cas.h1e_for_cas()
eri = pyscf.ao2mo.full(m, mf.mo_coeff[:, ncore:ncore+ncas], aosym='1').reshape(ncas, ncas, ncas, ncas)
np.save(f'{base}_{basis}_ecore', ecore)
np.save(f'{base}_{basis}_h1', h1)
np.save(f'{base}_{basis}_eri', eri)

e_tot, e_cas, ci, mo_coeff, mo_energy = cas.kernel(verbose=5)
print(f'{e_tot=}')
np.save(f'{base}_{basis}_civec', ci)
