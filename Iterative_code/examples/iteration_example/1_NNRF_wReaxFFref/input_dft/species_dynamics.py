from ase import Atoms, Atom
from ase.io import read, write
from ase.calculators.vasp import Vasp
from ase.db import connect


atoms = read('POSCAR_1')
calc = Vasp(xc = 'pbe',
			lcharg = False,
			lwave = False,
			lorbit = 11,
			potim = 1.0,
			encut = 400,
			ibrion = 0,
			nsw = 1000,
			kpts = [1,1,1],
			ivdw = 10,
			ismear = 0,
			sigma = 0.2,
			ediff = 1e-5,
			ediffg = -0.03,
			nelm = 500,
			algo = 'Fast',
			prec = 'Accurate',
			isif = 2,
			isym = 0,
			tebeg = 500,
			teend = 500,
			smass = -1,
			ncore = 32,)
atoms.set_calculator(calc)
atoms.get_potential_energy()
