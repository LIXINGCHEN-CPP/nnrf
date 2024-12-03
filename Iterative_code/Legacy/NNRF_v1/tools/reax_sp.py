#!/bin/env python

from subprocess import call
from lammpslib import write_lammps_data
from ase.io import read, write
from ase import Atoms, Atom
import numpy as np
import sys, json, os, collections, glob
import pandas as pd
from ase.db import connect
from ase.units import eV, kcal, mol


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


def reax_input_botable(filename,elements,ffield):
	e_string = ""
	for e in elements:
		e_string += "%s " % e

	with open(filename, 'w') as f:
		strings = "# Initialization\n"
		strings += "units           real\n"
		strings += "boundary        p p p\n"
		strings += "atom_style      charge\n"
		strings += "neighbor        2.0 nsq\n"
		strings += "neigh_modify    delay 2\n"
		strings += "read_data       ${filename}.lmp\n"
		strings += "# Potential Parameters\n"
		strings += "pair_style      reax/c NULL safezone 4.0 mincap 400\n"
		strings += "pair_coeff      * * %s %s\n" % (ffield, e_string)
		strings += "compute reax all pair reax/c\n"
		strings += "fix             10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n"
		strings += "variable eb equal c_reax[1]\n"
		strings += "variable ea equal c_reax[2]\n"
		strings += "variable elp equal c_reax[3]\n"
		strings += "variable emol equal c_reax[4]\n"
		strings += "variable ev equal c_reax[5]\n"
		strings += "variable epen equal c_reax[6]\n"
		strings += "variable ecoa equal c_reax[7]\n"
		strings += "variable ehb equal c_reax[8]\n"
		strings += "variable et equal c_reax[9]\n"
		strings += "variable eco equal c_reax[10]\n"
		strings += "variable ew equal c_reax[11]\n"
		strings += "variable ep equal c_reax[12]\n"
		strings += "variable efi equal c_reax[13]\n"
		strings += "variable eqeq equal c_reax[14]\n"
		strings += "thermo_style    custom step v_eb v_ea v_elp v_emol v_ev v_epen v_ecoa v_ehb v_et v_eco v_ew v_ep v_efi v_eqeq press pe\n"
		strings += "thermo          1\n"
		strings += "fix             rbo all reax/c/bonds 1 bonds.reaxc_${filename}\n"
		strings += "dump            min all custom 1 min.dump_${filename} id element x y z q fx fy fz\n"
		strings += "dump_modify     min element %s\n" % e_string
		strings += "min_style cg\n"
		strings += "minimize 1e-8 1e-8  0  0\n"
		strings += "undump          min\n"
		strings += "run             0\n"
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
				
		if species.count(0) < int(0.95*len(lines)/2):
			collect[mol] = species
			test[mol] = species
	data2 = pd.DataFrame(data=collect)
	data2.to_csv('all_3.csv')

def Reaction_Analysis(md_dir):

	return 



lmp_path = "/depot/lpl/apps/LAMMPS/lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/src/lmp_mpi"
elements = ['H','C','O','N']
xyzfile = sys.argv[1]


images = read(xyzfile, index=':')

cores = 1

bo_string = ""
dump_string = ""

sp_total = open('species.out','w')


for i in range(0,len(images),100):
	image = images[i]
	image.set_cell([11.42,10.58,13.14])
	syms = image.get_chemical_symbols()
	pos = image.get_positions()
	elements = ['H','C','O','N']
	D = {}
	for etype in elements:
		D[etype] = []

	for e,p in zip(syms,pos):
		for etype in elements:
			if e == etype:
				D[e].append(p)

	formula = ""
	positions = []
	for etype in elements:
		formula += etype + str(len(D[etype]))
		for pp in D[etype]:
			positions.append(pp)

	atoms = Atoms(formula,
			cell=image.cell,
			pbc=image.pbc,
			positions=positions)

	chem = atoms.get_chemical_symbols()
	elements = list(set(chem))
	atom_types = {}
	for el, j in zip(elements, range(len(elements))):
		atom_types[el] = j+1

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
