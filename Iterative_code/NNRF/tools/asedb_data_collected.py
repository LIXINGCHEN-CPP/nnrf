#!/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from ase.db import connect
from ase.units import eV, kcal, mol
import sys
import numpy as np
import glob

dbnames = glob.glob('*.db')
conversion_flag = sys.argv[1] # 1:eV 2:kcal/mol

for dbname in dbnames:
	print("Write %s" % dbname)
	db = connect(dbname)
	images = list(db.select())

	st = dbname.split(',')[0]
	xyzname = dbname.split('.')[0] + '.data'
	xyz = open(xyzname, 'w')
	s1 = []

	#Energy : kcal/mol
	#Force  : kcal/mol-Ang
	#Stress : GPa

	for i in range(len(images)):
		Nelements = len(images[i].positions)
		positions = images[i].positions

		if conversion_flag == '2':
			energy = images[i].energy * ( eV / (kcal/mol) )
		else:
			energy = images[i].energy
		energy = energy
		force = images[i].forces
		cell = images[i].cell
		cell_string = " ".join([str(c[0])+" "+str(c[1])+" "+str(c[2]) for c in cell])
		cell_string = '"' + cell_string +'"'
		stress = images[i].stress
		s1.append([tmp for tmp in stress])
		stress_string = " ".join(["%6.4f" % -s for s in stress])
		stress_string = '"' + stress_string +'"'
		xyz.write("begin\n")
		xyz.write("comment \n")
		xyz.write("lattice %20.10f %20.10f %20.10f\n" % (cell[0][0],cell[0][1],cell[0][2]))
		xyz.write("lattice %20.10f %20.10f %20.10f\n" % (cell[1][0],cell[1][1],cell[1][2]))
		xyz.write("lattice %20.10f %20.10f %20.10f\n" % (cell[2][0],cell[2][1],cell[2][2]))
		#xyz.write("config_type=%s energy=%s virial=%s Lattice=%s Properties=species:S:1:pos:R:3:force:R:3:Z:I:1\n" \
		#		% (st,str(energy),stress_string, cell_string))

		elements = images[i].symbols
		numbers = images[i].numbers

		elements_types = list(set(elements))
		for types in elements_types:
			for j in range(len(elements)):
				if elements[j] == types:
					e = elements[j]
					#p = positions[j] / cell[0][0]
					p = positions[j]
					p_string = ["%20.10f" % pp for pp in p]
					p_s = "".join(p_string)
					if conversion_flag == '2':
						f = force[j] * ( eV / (kcal/mol) )
					else:
						f = force[j]
					n = numbers[j]
					f_string = ["%20.10f" % ff for ff in f]
					f_s = "".join(f_string)
					xyz.write("atom %s %s %20.10f %20.10f %s\n" %(p_s, e, 0.0, 0.0, f_s))
		
		xyz.write("energy %20.10f\n" % energy)
		xyz.write("charge 0\n")
		xyz.write("end\n")

