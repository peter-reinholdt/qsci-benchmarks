#!/usr/bin/env python

import pyscf
import pyscf.fci
import pyscf.tools
import numpy as np
import sys
import os

norb = 20
nelec = (15,15)
h1 = pyscf.tools.fcidump.read('fe2s2')['H1']
h2 = pyscf.tools.fcidump.read('fe2s2')['H2']
eri = pyscf.ao2mo.restore('1', h2, norb=norb)

fcisolver = pyscf.fci.direct_spin0.FCISolver()
fcisolver.threads = 16
fcisolver.max_space = 20
fcisolver.max_cycle = 400
fcisolver.nroots = 1
fcisolver.kernel(h1, eri, norb, nelec, verbose=5)

np.save(f'civec', fcisolver.ci)

