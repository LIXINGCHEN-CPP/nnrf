#!/bin/env python

import sys, os, glob
import numpy as np
from ase.units import eV, kcal, mol
from ase.db import connect
from ase.io import read, write

def vasp2db(dbname, cal_dir, db_output='.', interval=1):

	asedb_name = db_output + "/" + dbname + '.db'
	if os.path.exists(asedb_name):
		print("asedb file already exist")
		answer = input("overwrite? [y/n]")
		if answer == "y":
			pass
		elif answer == "n":
			sys.exit()

	db = connect(asedb_name)

	pwd = os.getcwd()
	check_outcar = os.path.exists(pwd+'/OUTCAR')

	#Collect vasp OUTCARs into asedb

	if check_outcar == False:
		# collect single point calculations from numeric dir
		listdir = [d for d in os.listdir('.')]
		listdir = sorted([int(each) for each in listdir if each.isdigit()])

		
		for a in range(0, len(listdir), interval):
			i = listdir[a]
			subdir = "./"+str(i)
			ls = glob.glob(subdir+"/*")
			if subdir+"/fail" not in ls:
				images = read(subdir+"/OUTCAR",index=":")
				if len(images) == 1:
					atoms = images[0]
				else:
					atoms = images[-1]
			print("dir %d, PE %f eV" % (i, atoms.get_potential_energy()))
			db.write(atoms)

	else:
		# collect multiple SCF from vasp OUTCAR
		images = read('./OUTCAR',index=":")
		for i in range(len(images)):
			print("SCF %d, PE %f eV" % (i, images[i].get_potential_energy()))
			db.write(images[i])

	#Convert asedb into n2p2 data file
	# units : kcal/mol, kcal/mol/Å and GPa for energy, force, stress, respectively
	eVtoKcalpMol = (eV/(kcal/mol))
	print(eVtoKcalpMol)

	db = connect(asedb_name)
	images = list(db.select())
	n2p2db_name = db_output + "/" + dbname +'.data'
	n2p2db = open(n2p2db_name, 'w')
	s1 = []

	for i in range(len(images)):
		Natoms = len(images[i].positions)
		positions = images[i].positions

		energy = images[i].energy * eVtoKcalpMol
		force = images[i].forces
		cell = images[i].cell
		cell_string = " ".join([str(c[0])+" "+str(c[1])+" "+str(c[2]) for c in cell])
		cell_string = '"' + cell_string +'"'
		stress = images[i].stress
		s1.append([tmp for tmp in stress])
		stress_string = " ".join(["%6.4f" % -s for s in stress])
		stress_string = '"' + stress_string +'"'

		n2p2db.write("begin\n")
		n2p2db.write("comment\n")
		n2p2db.write("lattice %20.10f %20.10f %20.10f\n" % (cell[0][0],cell[0][1],cell[0][2]))
		n2p2db.write("lattice %20.10f %20.10f %20.10f\n" % (cell[1][0],cell[1][1],cell[1][2]))
		n2p2db.write("lattice %20.10f %20.10f %20.10f\n" % (cell[2][0],cell[2][1],cell[2][2]))

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
					f = force[j] * eVtoKcalpMol
					n = numbers[j]
					f_string = ["%20.10f" % ff for ff in f]
					f_s = "".join(f_string)
					n2p2db.write("atom %s %s %20.10f %20.10f %s\n" %(p_s, e, 0.0, 0.0, f_s))

		n2p2db.write("energy %20.10f\n" % energy)
		n2p2db.write("charge 0\n")
		n2p2db.write("end\n")
	n2p2db.close()



vasp2db(dbname='test', cal_dir='.', db_output='.', interval=1)
