from ase import Atoms, Atom
from ase.io import read, write
from ase.calculators.vasp import Vasp
from ase.db import connect


atoms = read('POSCAR_1')
calc = Vasp(xc = 'pbe',
			lcharg = False,
			lwave = False,
			lorbit = 11,
			potim = 0.05,
			encut = 500,
			ibrion = 1,
			nsw = 0,
			kpts = [1,1,1],
			ivdw = 10,
			ismear = 0,
			sigma = 0.2,
			ediff = 1e-5,
			ediffg = -0.03,
			nelm = 2000,
			algo = 'Normal',
			prec = 'Accurate',
			isif = 2,
			isym = 0,
			ncore = 32)
atoms.set_calculator(calc)
atoms.get_potential_energy()
