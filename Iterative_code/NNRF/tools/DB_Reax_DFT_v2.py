#!/bin/env python

from subprocess import call
import sys
import re
import json
from NNRF.lammpslib import write_lammps_data
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.io import read, write
import sys
import pandas as pd
from ase.db import connect
from ase.units import eV, kcal, mol
from ase import Atoms, Atom
from ase.data import atomic_numbers
from ase.visualize import view
import sys
import numpy as np
import collections

lmp_path = "/depot/lpl/apps/LAMMPS/lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/src/lmp_mpi"
atomsk_path="/depot/lpl/data/pilsun_files/SNAP/reax/atomsk/atomsk_b0.10.6_Linux-amd64/atomsk"


def atomsk(basename, xyz_filename):
	atomsk_command = atomsk_path +  " " + xyz_filename + " lmp -properties charge.txt"
	call(atomsk_command, shell=True)	    
	return

def lammps(lmp_filename):
	lammps_command ="mpiexec -np 1 " + lmp_path + " -var filename " + lmp_filename + " -in base.in > lmp.out"
	call(lammps_command, shell=True)
	return

def input_file(filename, elements):
	e_string = " ".join(elements)
	
	with open(filename,'w') as f:
		string = "#variable simname hmx_test"
		string += """
units           real
boundary        p p p
atom_style      charge
neighbor        2.0 nsq
neigh_modify    delay 2
box             tilt large
read_data       ${filename}.reax

#    Potential Parameters
pair_style      reax/c NULL  safezone 4.0 mincap 400
"""
		string += "pair_coeff      * * ffield " + e_string + "\n"
		string += """compute reax all pair reax/c
fix             10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c
#####################################################

variable eb equal c_reax[1]
variable ea equal c_reax[2]
variable elp equal c_reax[3]
variable emol equal c_reax[4]
variable ev equal c_reax[5]
variable epen equal c_reax[6]
variable ecoa equal c_reax[7]
variable ehb equal c_reax[8]
variable et equal c_reax[9]
variable eco equal c_reax[10]
variable ew equal c_reax[11]
variable ep equal c_reax[12]
variable efi equal c_reax[13]
variable eqeq equal c_reax[14]

thermo_style    custom step v_eb v_ea v_elp v_emol v_ev v_epen v_ecoa v_ehb v_et v_eco v_ew v_ep v_efi v_eqeq pe pxx pyy pzz pxy pxz pyz
thermo   1

dump            d1 all custom 1 force.dump fx fy fz
dump_modify     d1 sort id

run           0"""
		f.write(string)

	return


dbname = sys.argv[1]
db = connect(dbname)
images = list(db.select())


tmp_name = dbname[:-3]
#xyzname = tmp_name + '.xyz'
#xyz = open(xyzname, 'w')

new_db = connect(tmp_name + '_ref.db')

s1 = []
for i in range(len(images)):
	st = "RDX"
	Nelements = len(images[i].positions)
	positions = images[i].positions
	energy = images[i].energy * ( eV / (kcal/mol) )
	force = images[i].forces
	cell = images[i].cell
	pbc = images[i].pbc
	cell_string = " ".join([str(c[0])+" "+str(c[1])+" "+str(c[2]) for c in cell])
	cell_string = '"' + cell_string +'"'
	stress = images[i].stress
	s1.append([tmp for tmp in stress])
	stress_string = " ".join([str(-s) for s in stress])
	stress_string = '"' + stress_string +'"'
	#xyz.write("%d\n" % Nelements)
	#xyz.write("config_type=%s energy=%s virial=%s Lattice=%s Properties=species:S:1:pos:R:3:force:R:3:Z:I:1\n" \
    #			% (st,str(energy),stress_string, cell_string))
	elements = images[i].symbols
	counter = collections.Counter(elements)
	element_short = list(dict.fromkeys(elements))
	element_ordered = list(sorted(element_short,key=lambda e:atomic_numbers[e]))
	input_file('base.in',element_ordered)
	numbers = images[i].numbers

	#ref_ref = 16.24*counter['H'] + 81.80*counter['C'] + 37.42*counter['O'] + 52.79*counter['N']
	#dft_ref = -77.89*counter['H'] + -213.68*counter['C'] + -113.47*counter['O'] + -191.42*counter['N']
	ref_ref = 0
	dft_ref = 0

	element_ordered = []
	positions_ordered = []
	forces_ordered = []
	for types in element_short:
		for j in range(len(elements)):
			if elements[j] == types:
				e = elements[j]
				element_ordered.append(e)
				p = positions[j]
				positions_ordered.append(p)
				p_string = ["%14.6f" % pp for pp in p]
				p_s = "".join(p_string)
				f = force[j] * ( eV / (kcal/mol) )
				forces_ordered.append(f)
				n = numbers[j]
				f_string = ["%14.6f" % ff for ff in f]
				f_s = "".join(f_string)
				#xyz.write("%s %s %s %d\n" %(e, p_s, f_s, n))
	formula = ""
	for types in element_short:
		formula += types + str( element_ordered.count(types) )
	atoms = Atoms(formula, cell=cell, pbc=pbc, positions=positions_ordered)
	chem = atoms.get_chemical_symbols()
	elements = list(dict.fromkeys(chem))
	elements_o = list(sorted(elements, key=lambda e: atomic_numbers[e]))
	atom_types = {}
	for el, j in zip(elements_o, range(len(elements_o))):
		atom_types[el] = elements.index(el) + 1

	#view(atoms)
	#print(atoms[0])
	#print(atoms[1])
	basename = str(i+1)
	xyz_filename = basename + '.reax'
	write_lammps_data(filename=xyz_filename, 
			atoms=atoms, 
			atom_types=atom_types, units='real')
	atomsk(basename, xyz_filename)
	lammps(basename)

	#read forces
	ref_force = np.loadtxt('force.dump', skiprows=9)
	dft_force = np.array([list(f) for f in forces_ordered])
	
	#read energy
	dft_en = energy
	out = open('lmp.out', 'r')
	call('cp lmp.out lmp_%d.out' % (i+1), shell=True)
	lines = out.readlines()
	for m in range(len(lines)):
		line = lines[m].split()
		if line[0] == 'Step':
			index = line.index('PotEng')
			thermout = lines[m+1].split()
			ref_en = float(thermout[index])
			index = line.index('Pxx')
			press = thermout[index:]
			ref_press = np.array([float(p) for p in press]) * 0.00010132501
			break

	dft_press = stress
	#DFT - REAX
	en = (dft_en-dft_ref) - (ref_en-ref_ref)
	print(dft_en, dft_ref, ref_en, ref_ref)
	en = en * (kcal/mol/eV)
	#print(dft_en-dft_ref, ref_en-ref_ref)
	factor = (kcal/mol/eV)
	#print(dft_press, ref_press)
	press = np.array(dft_press) - np.array(ref_press)
	#print(press)
	force = [np.array(dft_force[i])*factor-np.array(ref_force[i])*factor for i in range(len(dft_force))]
	new_atoms = Atoms(formula, pbc=pbc,cell=cell,positions=positions_ordered)
	new_atoms.set_calculator(SPC(new_atoms,energy=en,forces=force,stress=press))
	new_db.write(new_atoms)


call('rm lmp_* *.lmp *.reax force.dump lmp.out min.xyz', shell=True)





	

