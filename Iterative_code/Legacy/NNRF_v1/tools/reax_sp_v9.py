#!/bin/env python
from subprocess import call
from ase.calculators.lammpslib import write_lammps_data
from ase.io import read, write
from ase import Atoms, Atom
import numpy as np
import sys, json, os, collections, glob
import pandas as pd
from ase.db import connect
from ase.units import eV, kcal, mol
from ase.data import atomic_numbers


def read_MD_energy(lammps_out):
	md_out = open(lammps_out, "r")
	md_out_contents = md_out.readlines()
	for i in range(len(md_out_contents)):
		if 'PotEng' in md_out_contents[i]:
			outline = i+1
	line1 = md_out_contents[outline-1].split()
	index = line1.index('PotEng')
	line2 = md_out_contents[outline].split()
	md_energy = line2[index]

	return md_energy

def read_DFT_data(dft_db,elements):
	#dftDB is asedb
	dftDB = connect(dft_db)
	DB = list(dftDB.select())
	dft_energies = []
	dft_positions = []
	dft_forces = []
	for i in range(len(DB)):
		dft_symbols = DB[i].symbols
		dft_energy = DB[i].energy * ( eV / (kcal/mol) )
		dft_force = DB[i].forces * ( eV / (kcal/mol) )
		dft_position = DB[i].positions

		dft_symbols2 = []
		dft_force2 = []
		dft_position2 = []

		for e in elements:
			for j in range(len(dft_symbols)):
				if e == dft_symbols[j]:
					f = list(dft_force[j])
					p = list(dft_position[j])
					dft_symbols2.append(e)
					dft_force2.append(f)
					dft_position2.append(p)

		dft_energies.append(dft_energy)
		dft_forces.append(dft_force2)
		dft_positions.append(dft_position2)
	return dft_energies,dft_positions,dft_forces

def read_BondOrder_new(BO_filename):
	#read output file of lammps BO table
	BO_filename = BO_filename
	bo_file = open(BO_filename,"r")
	bo_contents = bo_file.readlines()
	bos = []
	Ntraj = 0
	natoms = int(bo_contents[2].split()[-1])
	condition = True
	while condition:
		if Ntraj == 0:
			bo_inf = bo_contents[0:natoms+7]
		else:
			bo_inf = bo_contents[Ntraj*(natoms+8):(Ntraj+1)*(natoms+8)]
		if len(bo_inf) == 0: break
		timestep=bo_inf[0]
		natoms=int(bo_inf[2].split()[-1])
		bo = []
		bo_inf2 = bo_inf[7:natoms+7]
		for j in range(len(bo_inf2)):
			atom_info = bo_inf2[j].split()
			atom_index=int(atom_info[0]); atom_type=atom_info[1]; nn=atom_info[2]
			nn_index=atom_info[3:3+int(nn)]; nn_bo=atom_info[3+1+int(nn):3+1+int(nn)*2]
			nn = int(nn)
			nn_index = [int(k) for k in nn_index]
			nn_bo = [float(k) for k in nn_bo]
			tbo=float(atom_info[3+1+int(nn)*2])
			bo_tmp=[atom_index,nn_index,nn_bo,tbo]
			bo.append(bo_tmp)
		bo = sorted(bo, key=lambda x: x[0])
		Ntraj += 1
		bos.append(bo)
	return bos

def read_MD_dump(dump_filename,elements):
	#dump_format = 'id element x y z q fx fy fz'
	md_dump = open(dump_filename, "r")
	dump_contents_all = md_dump.readlines()
	natoms = int(dump_contents_all[3])
	Ntraj = 0
	md_symbols = []
	md_positions =[]
	md_forces = []
	while True:
		contents = dump_contents_all[Ntraj*natoms+9:(Ntraj+1)*natoms+9]
		if len(contents) == 0: break
		single_image = []
		md_symbol = []
		md_force = []
		md_position = []
		for i in range(len(contents)):
			line = contents[i].split()
			single_image.append([int(line[0])]+line[1:])
		single_image = sorted(single_image, key=lambda x: x[0])
		Ntraj += 1
		for i in range(len(single_image)):
			md_symbol.append(single_image[i][1])
			md_force.append( [ float(f) for f in single_image[i][6:9] ] )
			md_position.append( [ float(f) for f in single_image[i][2:5] ] )
		md_symbols.append(md_symbol)
		md_positions.append(md_position)
		md_forces.append(md_force)
	return md_symbols, md_positions, md_forces


def write_reax_input(filename):
	f = open(filename, 'w')
	strings = \
"""
#variable simname hmx_test

# Initialization
units           real
boundary        p p p

atom_style      charge
neighbor        2.0 nsq
neigh_modify    delay 2

read_data       ${filename}.lmp

#    Potential Parameters

#pair_style      reax/c NULL safezone 4.0 mincap 400
#pair_coeff      * * ffield_2014 H C O N

pair_style       hybrid/overlay snap reax/c NULL safezone 4.0 mincap 400
pair_coeff       * * reax/c ffield_2014 H C O N
pair_coeff       * * snap HCON.snapcoeff H C O N HCON.snapparam H C O N


compute reax all pair reax/c
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

thermo_style    custom step v_eb v_ea v_elp v_emol v_ev v_epen v_ecoa v_ehb v_et v_eco v_ew v_ep v_efi v_eqeq press pe
thermo          1
fix             rbo all reax/c/bonds 1 bonds.reaxc_${filename}

# Energy Minimization
#thermo_style    custom step temp pe press
#thermo          100

dump            min2 all custom 1 min.dump_${filename} id element x y z q fx fy fz
dump_modify     min2 element H C O N


min_style cg
minimize 1e-8 1e-8  0  0
undump          min2

run             0


"""
	f.write(strings)
	return

def species_analysis(species_out):
	all_species = []
	with open(species_out,'r') as sp2:
		lines = sp2.readlines()

	for i in range(0,len(lines),2):
		line1 = lines[i].replace('\t',' ').split()
		line2 = lines[i+1].replace('\t',' ').split()
		all_species += line1

	a = list(set(all_species))
	a.remove('No_Moles')
	a.remove('No_Specs')
	a.remove('#')
	a.remove('Timestep')

	collect = {}
	for mol in a:
		species = []
		test = {}
		for i in range(0,len(lines),2):
			line1 = lines[i].replace('\t',' ').split()[4:]
			line2 = lines[i+1].replace('\t',' ').split()[3:]

			for j in range(len(a)):
				if a[j] in line1 and a[j] == mol:
					index1 = line1.index(a[j])
					species.append(int(line2[index1]))
			if mol not in line1:
				species.append(0)
		#if mol_input == mol:
		#	print(mol)
		#	for i in species:
		#		print(i)

				
		if species.count(0) < int(0.95*len(lines)/2):
			collect[mol] = species
			test[mol] = species
	data2 = pd.DataFrame(data=collect)
	data2.to_csv('chem_species.csv')


lmp_path = "/depot/lpl/apps/LAMMPS/lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/src/lmp_mpi"
elements = ['H','C','N','O']
xyzfile = sys.argv[1]
images = read(xyzfile, index=':')

cores = 1

bo_string = ""
dump_string = ""

sp_total = open('species.out','w')


for i in range(0,len(images),2000):
	

	Nelements = len(images[i].get_positions())
	positions = images[i].get_positions()
	#force = images[i].forces
	cell = images[i].get_cell()
	pbc = [1,1,1]
	elements = images[i].get_chemical_symbols()
	counter = collections.Counter(elements)
	element_short = list(dict.fromkeys(elements))
	element_ordered = list(sorted(element_short,key=lambda e:atomic_numbers[e]))
	numbers = images[i].numbers

	element_ordered = []
	positions_ordered = []
	#forces_ordered = []
	for types in element_short:
		for j in range(len(elements)):
			if elements[j] == types:
				e = elements[j]
				element_ordered.append(e)
				p = positions[j]
				positions_ordered.append(p)
				p_string = ["%14.6f" % pp for pp in p]
				p_s = "".join(p_string)
				#f = force[j] * ( eV / (kcal/mol) )
				#forces_ordered.append(f)
				n = numbers[j]
				#f_string = ["%14.6f" % ff for ff in f]
				#f_s = "".join(f_string)
				#xyz.write("%s %s %s %d\n" %(e, p_s, f_s, n))
	formula = ""
	for types in element_short:
		formula += types + str( element_ordered.count(types) )
	atoms = Atoms(formula, cell=cell, pbc=pbc, positions=positions_ordered)
	chem = atoms.get_chemical_symbols()
	elements = list(dict.fromkeys(chem))
	elements_o = list(sorted(elements, key=lambda e: atomic_numbers[e]))
	print(elements_o)
	atom_types = {}
	for el, j in zip(elements_o, range(len(elements_o))):
		atom_types[el] = elements_o.index(el) + 1
	print(atom_types)
	 
	write_lammps_data(filename=str(i) + '.lmp',
			          atoms=atoms,
					  atom_types=atom_types,
					  units='real')
	lmp_filename = str(i)
	lammps_command = "mpiexec -np %d " % int(cores) +  lmp_path + " -var filename " + lmp_filename + " -in reax2.in > out_" + str(i)
	call(lammps_command, shell=True)

	#Write Json file
	bo_combined = 'bonds.reaxc'
	dump_combined = 'dump.out'

	mdout = 'out_%d' % i
	dumpout = 'min.dump_%d' % i
	bofile = 'bonds.reaxc_%d' % i
	species_out = 'species.out_%d' %i
	with open(species_out,'r') as species:
		spc_contents = species.readlines()
	sp_total.write(''.join(spc_contents))


	md_e = read_MD_energy(mdout)
	md_s, md_p, md_f = read_MD_dump(dumpout, elements)
	symbols = []
	types = md_s[0]
	for e in range(len(types)):
		symbols.append(types[e])
	bo = read_BondOrder_new(bofile)

	r_bo = open(bofile, 'r').readlines()
	t1 = r_bo[0].split()
	t1[2] = str(i) + "\n"
	r_bo[0] = " ".join(t1)
	r_dump = open(dumpout).readlines()
	r_dump[1] = str(i) + "\n"

	print('writing %d' % i )
	bo_string += " ".join(r_bo)
	dump_string += " ".join(r_dump)
	
	call('rm '+mdout+' '+dumpout+' '+bofile+' '+species_out+' *.lmp', shell=True)

sp_total.close()

with open('bonds.reaxc','w') as f1:
	f1.write(bo_string)
with open('min.dump','w') as f2:
	f2.write(dump_string)

species_analysis('species.out')
