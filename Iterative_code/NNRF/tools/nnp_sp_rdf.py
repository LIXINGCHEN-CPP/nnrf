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
import os

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

def read_MD_pressure(lammps_out):
	md_out = open(lammps_out, "r")
	md_out_contents = md_out.readlines()
	pr = ['Pxx','Pyy','Pzz','Pxy','Pxz','Pyz']
	press1 = []
	for p in pr:
		for i in range(len(md_out_contents)):
			if p in md_out_contents[i]:
				outline = i+1
		line1 = md_out_contents[outline-1].split()
		index = line1.index(p)
		line2 = md_out_contents[outline].split()
		press1.append(float(line2[index])*0.000101325)

	return press1



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
	collect = []
	while True:
		contents0 = dump_contents_all[Ntraj*(natoms+9):(Ntraj+1)*(natoms+9)]
		contents = contents0[9:]
		collect.append(contents0)

		Ntraj += 1
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
	final_dump = "".join(collect[-2])
	return np.array(md_symbols), np.array(md_positions), np.array(md_forces), final_dump

def write_reax_input(filename):
	f = open(filename, 'w')
	strings = "# Initialization\n"
	strings+= "units           real\n"
	strings+= "boundary        p p p\n"
	strings+= "atom_style      full\n"
	strings+= "neighbor        2.0 nsq\n"
	strings+= "neigh_modify    delay 2\n"
	strings+= "box tilt large\n"
	strings+= "read_data       ${filename}.lmp\n"
	strings+= "variable nnpCutoff       equal  5.0\n"
	strings+= "variable nnpDir          string 'nnp-data'\n"
	strings+= "variable mass_H          equal   1.0001\n"
	strings+= "variable mass_C          equal  12.0109\n"
	strings+= "variable mass_O          equal  15.9994\n"
	strings+= "variable mass_N          equal  14.0000\n"

	strings+= "mass 1 ${mass_H}\n"
	strings+= "mass 2 ${mass_C}\n"
	strings+= "mass 3 ${mass_O}\n"
	strings+= "mass 4 ${mass_N}\n"
	strings+= "#    Potential Parameters\n"
	strings+= "pair_style hybrid/overlay reax/c NULL nnp dir ${nnpDir} showew no showewsum 10000 resetew yes maxew 100000 cflength 1.0 cfenergy 1.0\n"
	strings+= "pair_coeff * * reax/c ffield_2014_ZeroF H C N O\n"
	strings+= "pair_coeff * * nnp ${nnpCutoff}\n"
	strings+= "compute reax all pair reax/c\n"
	strings+= "fix             10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n"
	strings+= "variable eb equal c_reax[1]\n"
	strings+= "variable ea equal c_reax[2]\n"
	strings+= "variable elp equal c_reax[3]\n"
	strings+= "variable emol equal c_reax[4]\n"
	strings+= "variable ev equal c_reax[5]\n"
	strings+= "variable epen equal c_reax[6]\n"
	strings+= "variable ecoa equal c_reax[7]\n"
	strings+= "variable ehb equal c_reax[8]\n"
	strings+= "variable et equal c_reax[9]\n"
	strings+= "variable eco equal c_reax[10]\n"
	strings+= "variable ew equal c_reax[11]\n"
	strings+= "variable ep equal c_reax[12]\n"
	strings+= "variable efi equal c_reax[13]\n"
	strings+= "variable eqeq equal c_reax[14]\n"
	#strings+= "thermo_style    custom step v_eb v_ea v_elp v_emol v_ev v_epen v_ecoa v_ehb v_et v_eco v_ew v_ep v_efi v_eqeq press pe\n"
	strings+= "thermo_style    custom step press pxx pyy pzz pxy pxz pyz pe\n"
	strings+= "thermo          1\n"
	strings+= "fix             rbo all reax/c/bonds 2 bonds.reaxc_${filename}\n"
	strings+= "dump            min2 all custom 2 min.dump_${filename} id type x y z q fx fy fz\n"
	strings+= "min_style cg\n"
	strings+= "minimize 1e-16 1e-16 0 0\n"
	strings+= "undump          min2\n"
	strings+= "run             1\n"

	f.write(strings)
	return


base_path = os.getcwd()

if os.path.exists('/scratch/brown'):
    n2p2_path = "/depot/lpl/apps/LAMMPS/7_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/n2p2_package"
    lmp_path = "/depot/lpl/apps/LAMMPS/7_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/src/lmp_mpi"

else:
    n2p2_path = "/depot/lpl/apps/LAMMPS/8_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132_rice/n2p2_package"
    lmp_path = "/depot/lpl/apps/LAMMPS/8_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132_rice/src/lmp_mpi"

atomsk_path="/depot/lpl/data/pilsun_files/SNAP/reax/atomsk/atomsk_b0.10.6_Linux-amd64/atomsk"

elements = ['H','C','N','O']
#xyzfile = sys.argv[1]
#images = read(xyzfile, index=':')
images = []
ndb = int(sys.argv[1])
for n in range(ndb):
	k = n+2
	filename = sys.argv[k]
	db = connect(filename)
	images += list(db.select())

cores = 1

bo_string = ""
dump_string = ""
final_dump_string = ""

md_energy = ""
forces = ""


element_list = {"H": {"mass": 1.001, "atomic_number": 1, "r1": 0.5, "w1": 1.0,"dft_ref":-76.481 , "ref_ref":-76.481 },\
	            "C": {"mass": 12.00, "atomic_number": 6, "r1": 0.5, "w1": 1.1,"dft_ref":-212.918, "ref_ref":-212.918},\
			 	"O": {"mass": 16.00, "atomic_number": 8, "r1": 0.5, "w1": 1.2,"dft_ref":-101.296, "ref_ref":-101.296},\
				"N": {"mass": 14.00, "atomic_number": 7, "r1": 0.5, "w1": 1.3,"dft_ref":-187.178, "ref_ref":-187.178}}

for i in range(0,len(images),1):
	image = images[i]
	syms = image.symbols
	pos = image.positions
	fos = image.forces
	Natoms = len(syms)
	#syms = image.get_chemical_symbols()
	#pos = image.get_positions()
	D = {}
	F = {}

	for etype in elements:
		D[etype] = []
		F[etype] = []

	for e,p,f in zip(syms,pos,fos):
		for etype in elements:
			if e == etype:
				D[e].append(p)
				F[e].append(f)

	dft_ref = 0
	ref_ref = 0

	for elm in syms:
		dft_ref += element_list[elm]["dft_ref"]
		ref_ref += element_list[elm]["ref_ref"]


	formula = ""
	positions = []
	forces_ = []
	for etype in elements:
		formula += etype + str(len(D[etype]))
		for pp in D[etype]:
			positions.append(pp)
		for ff in F[etype]:
			forces_.append(ff)

	forces_ = np.array(forces_)

	atoms = Atoms(formula,
			cell=image.cell,
			pbc=image.pbc,
			positions=positions)
	volume = atoms.get_volume() / len(atoms)

	####
	chem = atoms.get_chemical_symbols()
	atom_numbers = {}
	for el in elements:
		atom_numbers[el] = chem.count(el)
	#print(atom_numbers)
	atom_types = {}
	for el, j in zip(elements, range(len(elements))):
		atom_types[el] = j+1

	write_lammps_data(filename=str(i) + '.lmp',
			          atoms=atoms,
					  atom_types=atom_types,
					  units='real')
	lmp_filename = str(i)
	write_reax_input(filename='reax2.in')
	lammps_command = "mpiexec -np %d " % int(cores) +  lmp_path + " -var filename " + lmp_filename + " -in reax2.in > out_" + str(i)
	call(lammps_command, shell=True)

	#Write Json file
	bo_combined = 'bonds.reaxc'
	dump_combined = 'dump.out'

	mdout = 'out_%d' % i
	dumpout = 'min.dump_%d' % i
	bofile = 'bonds.reaxc_%d' % i

	md_e = read_MD_energy(mdout)
	#print(md_e)
	md_press = read_MD_pressure(mdout)
	print(md_press)

	md_s, md_p, md_f, final_dump = read_MD_dump(dumpout, elements)
	#print(final_dump)
	#print(md_f.shape)
	md_s = md_s[-1]
	md_p = md_p[-1]
	md_f = md_f[-1]
	#print(md_f.shape)


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


	nelement_string = ""
	for element in elements:
		na = atom_numbers[element]
		nelement_string += " %d " % na

	dft_e = image.energy * (eV/(kcal/mol))
	md_e = float(md_e)

	print((dft_e-dft_ref)/Natoms, (md_e-ref_ref)/Natoms)
	md_energy += nelement_string + "%12.4f %12.4f %12.4f\n" % (volume, (dft_e-dft_ref)/Natoms, (md_e-ref_ref)/Natoms)
	bo_string += " ".join(r_bo)
	dump_string += " ".join(r_dump)
	final_dump_string += final_dump

	dft_f_flatten = np.array(forces_ * eV/(kcal/mol)).flatten()
	md_f_flatten  = np.array(md_f).flatten()
	#print(dft_f_flatten.shape, md_f_flatten.shape)

	for f in range(len(dft_f_flatten)):
		forces += "%12.4f %12.4f\n" % (dft_f_flatten[f],md_f_flatten[f])



	if os.path.exists('0.lmp'):
		call('cp 0.lmp data', shell=True)
	#call('rm  '+dumpout+' '+bofile+' '+'*.lmp', shell=True)
	call('rm '+mdout+' '+dumpout+' '+bofile+' '+'*.lmp', shell=True)
	species_file = glob.glob("species.out_*")

with open('md_energy','w') as f1:
	f1.write(md_energy)
with open('bonds.reaxc','w') as f2:
	f2.write(bo_string)
with open('min.dump','w') as f3:
	f3.write(dump_string)
with open('forces','w') as f4:
	f4.write(forces)
with open('finals.dump','w') as f5:
	f5.write(final_dump_string)
