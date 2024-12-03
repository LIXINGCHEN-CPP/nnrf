#!/bin/env python
import subprocess, sys, json, os, collections, glob, random, copy
from ase.data import atomic_numbers, atomic_masses
from subprocess import call
from .lammpslib import write_lammps_data
from ase.io import read, write
from ase import Atoms, Atom
import numpy as np
import pandas as pd
from ase.db import connect
from ase.units import eV, kcal, mol
from ase.calculators.singlepoint import SinglePointCalculator as SPC

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
        press1.append(float(line2[index])*0.000101325) # check if this still works
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


def write_reax_min_input(filename, nnpDir, elements, reax_flag=False, reax_ref='ffield_2014_ZeroF', nnp_cutoff=5.0):
	e_string = " ".join(elements)

	f = open(filename, 'w')
	strings = "# Initialization\n"
	strings+= "units           real\n"
	strings+= "boundary        p p p\n"
	strings+= "atom_style      charge\n"
	strings+= "neighbor        2.0 nsq\n"
	strings+= "neigh_modify    delay 2\n"
	strings+= "box tilt large\n"
	strings+= "read_data       ${filename}\n"
	strings+= "variable nnpCutoff       equal   %f\n" % nnp_cutoff
	strings+= "variable nnpDir          string '%s'\n" % nnpDir
	strings+= "#    Potential Parameters\n"
	if reax_flag:
		strings+= "pair_style reax/c NULL\n"
		strings+= "pair_coeff * * reax/c %s %s\n" % (reax_ref, e_string)
	else:
		strings+= "pair_style hybrid/overlay reax/c NULL nnp dir ${nnpDir} showew no showewsum 10000 resetew yes maxew 100000 cflength 1.0 cfenergy 1.0\n"
		strings+= "pair_coeff * * reax/c %s %s\n" % (reax_ref, e_string)
		strings+= "pair_coeff * * nnp ${nnpCutoff}\n"
	strings+= "compute 		   reax all pair reax/c\n"
	strings+= "fix             10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n"
	#Add RDF analysis
	strings+= "compute         myRDF all rdf 200 1 1 1 2 1 3 1 4 2 1 2 2 2 3 2 4 3 1 3 2 3 3 3 4 4 1 4 2 4 3 4 4 cutoff 10.0\n"
	strings+= "fix             1 all ave/time 1 1 100 c_myRDF[*] file pair.rdf mode vector\n"
	strings+= "thermo_style    custom step press pxx pyy pzz pxy pxz pyz pe\n"
	strings+= "thermo          1\n"
	strings+= "fix             rbo all reax/c/bonds 2 bonds.reaxc\n"
	strings+= "fix             spe1 all reax/c/species 1 1 1 species.out element %s\n" % e_string
	strings+= "dump            min2 all custom 2 min.dump id type x y z q fx fy fz\n"
	strings+= "dump_modify     min2 element %s sort id\n" % e_string
	strings+= "min_style cg\n"
	strings+= "minimize 1e-16 1e-16 10000 10000\n"
	strings+= "undump          min2\n"
	strings+= "run             1\n"

	f.write(strings)
	return


def write_reax_input(filename, nnpDir, elements, reax_flag=False, reax_ref='ffield_2014_ZeroF', nnp_cutoff=5.0):
	e_string = " ".join(elements)

	f = open(filename, 'w')
	strings = "# Initialization\n"
	strings+= "units           real\n"
	strings+= "boundary        p p p\n"
	strings+= "atom_style      charge\n"
	strings+= "neighbor        2.0 nsq\n"
	strings+= "neigh_modify    delay 2\n"
	strings+= "box tilt large\n"
	strings+= "read_data       ${filename}\n"
	strings+= "variable nnpCutoff       equal   %f\n" % nnp_cutoff
	strings+= "variable nnpDir          string '%s'\n" % nnpDir
	strings+= "#    Potential Parameters\n"
	if reax_flag:
		strings+= "pair_style reax/c NULL\n"
		strings+= "pair_coeff * * reax/c %s %s\n" % (reax_ref, e_string)
	else:
		strings+= "pair_style hybrid/overlay reax/c NULL nnp dir ${nnpDir} showew no showewsum 10000 resetew yes maxew 100000 cflength 1.0 cfenergy 1.0\n"
		strings+= "pair_coeff * * reax/c %s %s\n" % (reax_ref, e_string)
		strings+= "pair_coeff * * nnp ${nnpCutoff}\n"
	strings+= "compute 		   reax all pair reax/c\n"
	strings+= "fix             10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n"
	strings+= "compute         peratom all pe/atom\n"

	#Add RDF analysis
	strings+= "compute         myRDF all rdf 200 1 1 1 2 1 3 1 4 2 1 2 2 2 3 2 4 3 1 3 2 3 3 3 4 4 1 4 2 4 3 4 4 cutoff 10.0\n"
	strings+= "fix             1 all ave/time 1 1 100 c_myRDF[*] file pair.rdf mode vector\n"
	strings+= "thermo_style    custom step press pxx pyy pzz pxy pxz pyz pe\n"
	strings+= "thermo          1\n"
	strings+= "fix             rbo all reax/c/bonds 2 bonds.reaxc\n"
	strings+= "fix             spe1 all reax/c/species 1 1 1 species.out element %s\n" % e_string
	strings+= "dump            min2 all custom 2 min.dump id type x y z q fx fy fz c_peratom[*]\n"
	strings+= "dump_modify     min2 element %s sort id\n" % e_string
	strings+= "min_style cg\n"
	strings+= "minimize 1e-16 1e-16 0 0\n"
	strings+= "undump          min2\n"
	strings+= "run             1\n"

	f.write(strings)
	return

def write_input(filename, nnpDir, elements, nnp_cutoff):
	e_string = " ".join(elements)

	f = open(filename, 'w')
	strings = "# Initialization\n"
	strings+= "units           real\n"
	strings+= "boundary        p p p\n"
	strings+= "atom_style      charge\n"
	strings+= "neighbor        2.0 nsq\n"
	strings+= "neigh_modify    delay 2\n"
	strings+= "box tilt large\n"
	strings+= "read_data       ${filename}\n"
	strings+= "variable nnpCutoff       equal  %f\n" % nnp_cutoff
	strings+= "variable nnpDir          string '%s'\n" % nnpDir
	strings+= "#    Potential Parameters\n"
	strings+= "pair_style nnp dir ${nnpDir} showew no showewsum 10000 resetew yes maxew 100000 cflength 1.0 cfenergy 1.0\n"
	strings+= "pair_coeff * * ${nnpCutoff}\n"
	strings+= "thermo_style    custom step press pxx pyy pzz pxy pxz pyz pe\n"
	strings+= "thermo          1\n"
	strings+= "dump            min2 all custom 2 min.dump id type x y z q fx fy fz\n"
	strings+= "dump_modify     min2 element %s sort id\n" % e_string
	strings+= "min_style cg\n"
	strings+= "minimize 1e-16 1e-16 0 0\n"
	strings+= "undump          min2\n"
	strings+= "run             1\n"

	f.write(strings)
	return


def reax_n2p2_minimize_str(nnps, atoms, commands, cores, reax, reax_flag, elements=[]):

	base_path = os.getcwd()
	n2p2_path = commands['n2p2']
	lmp_path = commands['lmp']
	atomsk_path = commands['atomsk']

	rmse_info = []
	if len(elements) == 0:
		elements= atoms.get_chemical_symbols()
		counter = collections.Counter(elements)

	element_short = list(dict.fromkeys(elements))
	element_ordered = list(sorted(element_short,key=lambda e:atomic_numbers[e]))

	chem = atoms.get_chemical_symbols()
	elements = list(dict.fromkeys(chem))
	if len(elements) == 0:
		elements_o = list(sorted(elements, key=lambda e: atomic_numbers[e]))
		atom_types = {}
		for el, j in zip(elements_o, range(len(elements_o))):
			atom_types[el]=elements.index(el) + 1
	else:
		elements_o = list(sorted(element_ordered, key=lambda e: atomic_numbers[e]))
		atom_types = {}
		for el, j in zip(elements_o, range(len(elements_o))):
			atom_types[el]=element_ordered.index(el) + 1

	lmp_filename = 'str.data'
	write_lammps_data(filename=lmp_filename,
					  atoms = atoms,
					  atom_types=atom_types,
					  units='real')

	#For loop for nnp for each structure
	nnp_e = []
	nnp_f = []
	reax_bo = []
	reax_species = []
	for nnpDir in nnps:
		nnp_name  = nnpDir.split('/')[-1]
		write_reax_min_input(filename='nnp_input.in', nnpDir=nnpDir, elements=element_ordered, reax_flag=reax_flag, reax_ref=reax, nnp_cutoff=5.0)
		lammps_command = "mpiexec -np %d " % int(cores) +  lmp_path + " -var filename " + lmp_filename + " -in nnp_input.in > out"
		call(lammps_command, shell=True)
		call('cp min.dump min.dump_%s' % nnp_name,shell=True)
		mdout = 'out'
		dumpout = 'min.dump'
		bofile = 'bonds.reaxc'
		species_out = 'species.out'
		with open(species_out,'r') as species:
			spc_contents = species.readlines()
		reax_species.append(spc_contents)
	
		md_e = float(read_MD_energy(mdout))
		md_s, md_p, md_f, final_dump = read_MD_dump(dumpout, elements)
		types = md_s[0]
		md_s = md_s[-1]
		md_p = md_p[-1]
		md_f = md_f[-1]
		
		nnp_e.append(md_e)
		nnp_f.append(md_f.flatten())

		symbols = []
		for e in range(len(types)):
			symbols.append(types[e])

		bo = read_BondOrder_new(bofile)
		reax_bo.append(bo)
		


	#Provide Energy and Forces for each structure from a single or multiple nnp
	return nnp_e, np.array(nnp_f), reax_species, reax_bo








def n2p2_single_point(nnps, atoms, commands, cores, elements=[]):

	base_path = os.getcwd()
	n2p2_path = commands['n2p2']
	lmp_path = commands['lmp']
	atomsk_path = commands['atomsk']

	rmse_info = []
	#For loop for structures
	#for i in range(len(images)):
	#atoms = images[i]

	if len(elements) == 0:
		elements= atoms.get_chemical_symbols()
		counter = collections.Counter(elements)

	element_short = list(dict.fromkeys(elements))
	element_ordered = list(sorted(element_short,key=lambda e:atomic_numbers[e]))

	chem = atoms.get_chemical_symbols()
	elements = list(dict.fromkeys(chem))
	if len(elements) == 0:
		elements_o = list(sorted(elements, key=lambda e: atomic_numbers[e]))
		atom_types = {}
		for el, j in zip(elements_o, range(len(elements_o))):
			atom_types[el]=elements.index(el) + 1
	else:
		elements_o = list(sorted(element_ordered, key=lambda e: atomic_numbers[e]))
		atom_types = {}
		for el, j in zip(elements_o, range(len(elements_o))):
			atom_types[el]=element_ordered.index(el) + 1

	lmp_filename = 'str.data'
	write_lammps_data(filename=lmp_filename,
					  atoms = atoms,
					  atom_types=atom_types,
					  units='real')

	#For loop for nnp for each structure
	nnp_e = []
	nnp_f = []
	for nnpDir in nnps:
		write_input(filename='nnp_input.in',nnpDir=nnpDir, elements=element_ordered, nnp_cutoff=6.000)
		lammps_command = "mpiexec -np %d " % int(cores) +  lmp_path + " -var filename " + lmp_filename + " -in nnp_input.in > out"
		call(lammps_command, shell=True)
		mdout = 'out'
		dumpout = 'min.dump'

		md_e = float(read_MD_energy(mdout))
		md_s, md_p, md_f, final_dump = read_MD_dump(dumpout, elements)
		md_s = md_s[-1]
		md_p = md_p[-1]
		md_f = md_f[-1]
		
		nnp_e.append(md_e)
		nnp_f.append(md_f.flatten())

	#Provide Energy and Forces for each structure from a single or multiple nnp
	return nnp_e, np.array(nnp_f)

def Search_str(nnps, db_file, commands, cores):

	#db_file is asedb
	db = connect(db_file)
	images = list(db.select())
	total_images = []
	for i in range(len(images)):
		image = images[i]
		atoms = image
		#prin(image.formula)
		atoms = Atoms(symbols=image.formula,positions=image.positions,cell=image.cell)
		chem = atoms.get_chemical_symbols()

		nnp_ermse, nnp_frmse = n2p2_single_point(nnps, atoms, commands, cores)

		ermse1 = np.sqrt( ( nnp_ermse[1] - nnp_ermse[0] ) ** 2 ) / len(atoms)
		ermse2 = np.sqrt( ( nnp_ermse[2] - nnp_ermse[0] ) ** 2 ) / len(atoms)
		frmse1 = np.sqrt( np.mean( ( nnp_frmse[1] - nnp_frmse[0] ) ** 2 ) )
		frmse2 = np.sqrt( np.mean( ( nnp_frmse[2] - nnp_frmse[0] ) ** 2 ) )
		c_1 = abs(ermse1-ermse2)
		c_2 = abs(frmse1-frmse2)
		print(c_1, c_2)
		images1 = []
		while len(images1) < 51: 
			chem_2 = copy.deepcopy(chem)
			atoms_2 = copy.deepcopy(atoms)
			random.shuffle(chem_2)
			atoms_2.set_chemical_symbols(chem_2)
			atoms_2.rattle(stdev=random.uniform(0.05, 0.20), seed=random.randint(0,10000))

			syms = "".join(chem_2)

			nnp_e, nnp_f = n2p2_single_point(nnps, atoms_2, commands, cores)
			#use 3 nnps...
			ermse1 = np.sqrt( ( nnp_e[1] - nnp_e[0] ) ** 2 ) / len(atoms_2)
			ermse2 = np.sqrt( ( nnp_e[2] - nnp_e[0] ) ** 2 ) / len(atoms_2)
			frmse1 = np.sqrt( np.mean( ( nnp_f[1] - nnp_f[0] ) ** 2 ) ) 
			frmse2 = np.sqrt( np.mean( ( nnp_f[2] - nnp_f[0] ) ** 2 ) ) 
			print(abs( ermse1 - ermse2 ), abs( frmse1 - frmse2 ))
			if (abs( ermse1 - ermse2 ) > c_1) or (abs( frmse1 - frmse2 ) > c_2):
				images1.append(atoms_2)

			print(len(images1))
		write(db_file[:-3]+'_%d.xyz' % (i+1), images1, format='xyz')
		
		total_images += images1
	
	return total_images

def nnp_parity_CCAs(nnps, db_file, commands, cores, interval=1, elements=[]):

	#db_file is asedb
	db = connect(db_file)
	images = list(db.select())
	es = []
	fs = []
	for i in range(0,len(images),interval):
		print("%d calculating" %i)
		image = images[i]
		dft_e = image.energy *eV / ( kcal / mol ) 
		element1 = list(dict.fromkeys(list(sorted(image.symbols))))
		img_sym = image.symbols
		img_pos = image.positions
		img_for = image.forces
		s = []
		p = []
		f = []
		for e in element1:
			for j in range(len(img_sym)):
				if e == img_sym[j]:
					s.append(img_sym[j])
					p.append(img_pos[j])
					f.append(img_for[j])
		s = np.array(s)
		p = np.array(p)
		f = np.array(f)  * eV / ( kcal / mol )
		#for k in range(len(s)):
		#	print(s[k], p[k], f[k])


		atoms = Atoms(symbols=s,positions=p,cell=image.cell)
		atoms.set_calculator(SPC(atoms, energy=image.energy, forces=f))
		pos1  = atoms.get_positions()
		vol   = atoms.get_volume() / len(atoms)
		pos2  = image.positions
		chem = atoms.get_chemical_symbols()
		if len(elements) == 0:
			element2 = list(dict.fromkeys(chem))
		else:
			element2 = elements
		#print(element2)
		nnp_e, nnp_f = n2p2_single_point(nnps, atoms, commands, cores, elements)	

		#for k in range(len(atoms)):
		#	print(atoms[k], s[k])

		f_sorted_fl = f.flatten()
		f_sorted_fl = np.expand_dims(f_sorted_fl,axis=(0))

		dft_e = dft_e / len(atoms)
		nnp_e = nnp_e[0] / len(atoms)
		nnp_f1 = np.concatenate((f_sorted_fl, nnp_f), axis=0)
		nnp_f2 = np.transpose(nnp_f1)
		es.append([vol, dft_e] + [nnp_e])
		#fs.append(nnp_f2)
		fs += list(nnp_f2)


	es2 = np.array(es)
	fs2 = np.array(fs)
	np.savetxt('energy.txt',es2)
	np.savetxt('forces.txt',fs2)
	return es2, fs2

def nnp_parity_wReaxFFref(nnps, db_file, commands, cores, interval, reax, reax_flag, elements=[]):

	db_file_ext = db_file.split('.')[-1]
	if db_file_ext == 'db':
		#db_file is asedb
		db = connect(db_file)
		images = list(db.select())
		es = []
		fs = []
		for i in range(0,len(images),interval):
			print("%d calculating" %i)
			image = images[i]
			dft_e = image.energy *eV / ( kcal / mol ) 
			element1 = list(dict.fromkeys(list(sorted(image.symbols))))
			img_sym = image.symbols
			img_pos = image.positions
			img_for = image.forces
			s = []
			p = []
			f = []
			for e in element1:
				for j in range(len(img_sym)):
					if e == img_sym[j]:
						s.append(img_sym[j])
						p.append(img_pos[j])
						f.append(img_for[j])
			s = np.array(s)
			p = np.array(p)
			f = np.array(f)  * eV / ( kcal / mol )

			atoms = Atoms(symbols=s,positions=p,cell=image.cell)
			atoms.set_calculator(SPC(atoms, energy=image.energy, forces=f))
			pos1  = atoms.get_positions()
			vol   = atoms.get_volume() / len(atoms)
			pos2  = image.positions
			chem = atoms.get_chemical_symbols()
			if len(elements) == 0:
				element2 = list(dict.fromkeys(chem))
			else:
				element2 = elements
			nnp_e, nnp_f, reax_species, reax_bo, min_dump, pair_rdf = reax_n2p2_single_point(nnps, atoms, commands, cores, reax, reax_flag, elements)	
			#print(reax_species, reax_bo)
			f_sorted_fl = f.flatten()
			f_sorted_fl = np.expand_dims(f_sorted_fl,axis=(0))

			dft_e2 = dft_e / len(atoms)
			nnp_e = [nne/len(atoms) for nne in nnp_e]
			nnp_f1 = np.concatenate((f_sorted_fl, nnp_f), axis=0)
			nnp_f2 = np.transpose(nnp_f1)
			print(dft_e, dft_e2, nnp_e)
			es.append([vol, dft_e2] + nnp_e)
			fs.append(nnp_f2)
		es2 = np.array(es)
		fs2 = np.array(fs)
		fs2 = fs2.reshape(-1, fs2.shape[-1])
		np.savetxt('energy.txt',es2)
		np.savetxt('forces.txt',fs2)
		return es2, fs2

	elif db_file_ext == 'xyz':
		xyz = read(db_file, index=':')
		images = list(xyz)

		es = []
		fs = []
		es_files = []
		fs_files = []
		dump_series = []
		min_dump = []
		for i in range(0,len(images),interval):
			print("%d calculating" %i)
			image = images[i]
			element1 = list(dict.fromkeys(list(sorted(image.get_chemical_symbols()))))
			img_sym = image.get_chemical_symbols()
			img_pos = image.get_positions()
			s = []
			p = []
			for e in element1:
				for j in range(len(img_sym)):
					if e == img_sym[j]:
						s.append(img_sym[j])
						p.append(img_pos[j])
			s = np.array(s)
			p = np.array(p)

			atoms = Atoms(symbols=s,positions=p,cell=image.cell)
			pos1  = atoms.get_positions()
			vol   = atoms.get_volume() / len(atoms)
			pos2  = image.positions
			chem = atoms.get_chemical_symbols()
			if len(elements) == 0:
				element2 = list(dict.fromkeys(chem))
			else:
				element2 = elements

			nnp_e, nnp_f, reax_species, reax_bo, min_dump, pair_rdf = reax_n2p2_single_point(nnps, atoms, commands, cores, reax, reax_flag, elements)	
			min_dump_2 = [m+'_%d'%i  for m in min_dump]
			pair_rdf_2 = [r+'_%d'%i  for r in pair_rdf]
			for k in range(len(min_dump)):
				subprocess.call('mv %s %s' % (min_dump[k], min_dump_2[k]),shell=True)
			for k in range(len(pair_rdf)):
				subprocess.call('mv %s %s' % (pair_rdf[k], pair_rdf_2[k]),shell=True)

			#f_sorted_fl = f.flatten()
			#dft_e = dft_e / len(atoms)
			nnp_e = [nne/len(atoms) for nne in nnp_e]
			f_sorted_fl = np.zeros(len(nnp_f[0]))
			f_sorted_fl = np.expand_dims(f_sorted_fl,axis=(0))
		
			nnp_f1 = np.concatenate((f_sorted_fl,nnp_f), axis=0)
			nnp_f2 = np.transpose(nnp_f1)
			np.savetxt('energy_%d.txt' % i, np.array([[vol]+nnp_e]))
			np.savetxt('forces_%d.txt' % i,nnp_f2)
			es_files.append('energy_%d.txt' % i)
			fs_files.append('forces_%d.txt' % i)
			
		
		for l in range(len(min_dump)):
			min_files = [min_dump[l]+'_%d'%i for i in range(0,len(images),interval)]
			cat_com = 'cat '+' '.join(min_files) + ' > '+min_dump[l]+'_all'
			if os.path.exists(min_dump[l]+'_all'):
				subprocess.call('rm '+min_dump[l]+'_all',shell=True)
			subprocess.call(cat_com,shell=True)
			subprocess.call('rm '+' '.join(min_files),shell=True)
		#es2 = np.array(es)
		#fs2 = np.array(fs)
		#fs2 = fs2.reshape(-1, fs2.shape[-1])
		#np.savetxt('energy.txt',es2)
		#np.savetxt('forces.txt',fs2)
		subprocess.call('cat '+" ".join(es_files)+' > energy.txt',shell=True)
		subprocess.call('cat '+" ".join(fs_files)+' > forces.txt',shell=True)
		es2 = 'energy.txt'
		fs2 = 'forces.txt'

		return es2, fs2


def nnp_parity_wRef(nnps, db_file, commands, cores, interval, input_file, elements=[], total=None):

	db_file_ext = db_file.split('.')[-1]
	if db_file_ext == 'db':
		#db_file is asedb
		db = connect(db_file)
		images = list(db.select())
		es = []
		fs = []
		es_files = []
		fs_files = []
		dump_series = []
		min_dump = []
		if total != None:
			length = total
		else:
			length = len(images)
		for i in range(0,length,interval):
			print("%d calculating" %i)
			image = images[i]
			dft_e = image.energy *eV / ( kcal / mol ) 
			element1 = list(dict.fromkeys(list(sorted(image.symbols))))
			img_sym = image.symbols
			img_pos = image.positions
			img_for = image.forces
			s = []
			p = []
			f = []
			for e in element1:
				for j in range(len(img_sym)):
					if e == img_sym[j]:
						s.append(img_sym[j])
						p.append(img_pos[j])
						f.append(img_for[j])
			s = np.array(s)
			p = np.array(p)
			f = np.array(f)  * eV / ( kcal / mol )

			atoms = Atoms(symbols=s,positions=p,cell=image.cell)
			atoms.set_calculator(SPC(atoms, energy=image.energy, forces=f))
			pos1  = atoms.get_positions()
			vol   = atoms.get_volume() / len(atoms)
			pos2  = image.positions
			chem = atoms.get_chemical_symbols()
			if len(elements) == 0:
				element2 = list(dict.fromkeys(chem))
			else:
				element2 = elements
			nnp_e, nnp_f, min_dump  = Ref_n2p2_single_point(nnps, atoms, commands, cores, input_file, elements)	
			min_dump_2 = [m+'_%d'% i  for m in min_dump]

			for k in range(len(min_dump)):
				subprocess.call('mv %s %s' % (min_dump[k], min_dump_2[k]),shell=True)
				
			f_sorted_fl = f.flatten()
			f_sorted_fl = np.expand_dims(f_sorted_fl,axis=(0))

			dft_e = dft_e / len(atoms)
			nnp_e = [nne/len(atoms) for nne in nnp_e]
			nnp_f1 = np.concatenate((f_sorted_fl, nnp_f), axis=0)
			nnp_f2 = np.transpose(nnp_f1)
			#print([dft_e] + nnp_e)
			#es.append([vol, dft_e] + nnp_e)
			#fs.append(nnp_f2)
			np.savetxt('energy_%d.txt' % i, np.array([[vol, dft_e]+nnp_e]))
			np.savetxt('forces_%d.txt' % i,nnp_f2)
			es_files.append('energy_%d.txt' % i)
			fs_files.append('forces_%d.txt' % i)

		for l in range(len(min_dump)):
			min_files = [min_dump[l]+'_%d'%i for i in range(0,len(images),interval)]
			cat_com = 'cat '+' '.join(min_files) + ' > '+min_dump[l]+'_all'
			if os.path.exists(min_dump[l]+'_all'):
				subprocess.call('rm '+min_dump[l]+'_all',shell=True)
			subprocess.call(cat_com,shell=True)
			subprocess.call('rm '+' '.join(min_files),shell=True)
		#es2 = np.array(es)
		#fs2 = np.array(fs)
		#fs2 = fs2.reshape(-1, fs2.shape[-1])
		#np.savetxt('energy.txt',es2)
		#np.savetxt('forces.txt',fs2)
		subprocess.call('cat '+" ".join(es_files)+' > energy.txt',shell=True)
		subprocess.call('cat '+" ".join(fs_files)+' > forces.txt',shell=True)
		es2 = 'energy.txt'
		fs2 = 'forces.txt'

		return es2, fs2

	elif db_file_ext == 'xyz':
		xyz = read(db_file, index=':')
		images = list(xyz)

		es = []
		fs = []
		es_files = []
		fs_files = []
		dump_series = []
		min_dump = []
		if total != None:
			length = len(images)
		else:
			length = total
		for i in range(0,length,interval):
			print("%d calculating" %i)
			image = images[i]
			element1 = list(dict.fromkeys(list(sorted(image.get_chemical_symbols()))))
			img_sym = image.get_chemical_symbols()
			img_pos = image.get_positions()
			s = []
			p = []
			for e in element1:
				for j in range(len(img_sym)):
					if e == img_sym[j]:
						s.append(img_sym[j])
						p.append(img_pos[j])
			s = np.array(s)
			p = np.array(p)

			atoms = Atoms(symbols=s,positions=p,cell=image.cell)
			pos1  = atoms.get_positions()
			vol   = atoms.get_volume() / len(atoms)
			pos2  = image.positions
			chem = atoms.get_chemical_symbols()
			if len(elements) == 0:
				element2 = list(dict.fromkeys(chem))
			else:
				element2 = elements
			nnp_e, nnp_f, min_dump  = Ref_n2p2_single_point(nnps, atoms, commands, cores, input_file, elements)	
			min_dump_2 = [m+'_%d'%i  for m in min_dump]
			for k in range(len(min_dump)):
				subprocess.call('mv %s %s' % (min_dump[k], min_dump_2[k]),shell=True)
			#f_sorted_fl = f.flatten()
			#dft_e = dft_e / len(atoms)
			nnp_e = [nne/len(atoms) for nne in nnp_e]
			f_sorted_fl = np.zeros(len(nnp_f[0]))
			f_sorted_fl = np.expand_dims(f_sorted_fl,axis=(0))
		
			nnp_f1 = np.concatenate((f_sorted_fl,nnp_f), axis=0)
			nnp_f2 = np.transpose(nnp_f1)
			print(nnp_e)
			#es.append([vol] + nnp_e)
			#fs.append(nnp_f2)
			np.savetxt('energy_%d.txt' % i, np.array([[vol]+nnp_e]))
			np.savetxt('forces_%d.txt' % i,nnp_f2)
			es_files.append('energy_%d.txt' % i)
			fs_files.append('forces_%d.txt' % i)
		
		for l in range(len(min_dump)):
			min_files = [min_dump[l]+'_%d'%i for i in range(0,len(images),interval)]
			cat_com = 'cat '+' '.join(min_files) + ' > '+min_dump[l]+'_all'
			if os.path.exists(min_dump[l]+'_all'):
				subprocess.call('rm '+min_dump[l]+'_all',shell=True)
			subprocess.call(cat_com,shell=True)
			subprocess.call('rm '+' '.join(min_files),shell=True)
		#es2 = np.array(es)
		#fs2 = np.array(fs)
		#fs2 = fs2.reshape(-1, fs2.shape[-1])
		#np.savetxt('energy.txt',es2)
		#np.savetxt('forces.txt',fs2)
		subprocess.call('cat '+" ".join(es_files)+' > energy.txt',shell=True)
		subprocess.call('cat '+" ".join(fs_files)+' > forces.txt',shell=True)
		es2 = 'energy.txt'
		fs2 = 'forces.txt'

		return es2, fs2


def Ref_n2p2_single_point(nnps, atoms, commands, cores, input_file, elements=[]):

	base_path = os.getcwd()
	n2p2_path = commands['n2p2']
	lmp_path = commands['lmp']
	atomsk_path = commands['atomsk']

	rmse_info = []
	#For loop for structures
	#for i in range(len(images)):
	#atoms = images[i]

	if len(elements) == 0:
		elements= atoms.get_chemical_symbols()
		counter = collections.Counter(elements)

	element_short = list(dict.fromkeys(elements))
	element_ordered = list(sorted(element_short,key=lambda e:atomic_numbers[e]))

	chem = atoms.get_chemical_symbols()
	elements = list(dict.fromkeys(chem))
	if len(elements) == 0:
		elements_o = list(sorted(elements, key=lambda e: atomic_numbers[e]))
		atom_types = {}
		for el, j in zip(elements_o, range(len(elements_o))):
			atom_types[el]=elements.index(el) + 1
	else:
		elements_o = list(sorted(element_ordered, key=lambda e: atomic_numbers[e]))
		atom_types = {}
		for el, j in zip(elements_o, range(len(elements_o))):
			atom_types[el]=element_ordered.index(el) + 1

	lmp_filename = 'str.data'
	cell_la = atoms.get_cell_lengths_and_angles()
	print(sum(cell_la[3:]))
	if sum(cell_la[3:]) == 270.0:
		triclinic = False
	else:
		triclinic = True
	write_lammps_data(filename=lmp_filename,
					  atoms = atoms,
					  atom_types=atom_types,
					  units='real',triclinic=triclinic)


	#For loop for nnp for each structure
	nnp_e = []
	nnp_f = []
	min_dump = []
	for nnpDir in nnps:
		lammps_command = "mpiexec -np %d " % int(cores) +  lmp_path + " -var nnpDir " + nnpDir +" -in %s > out" % input_file
		call(lammps_command, shell=True)
		mdout = 'out'
		dumpout = 'min.dump'
	
		md_e = float(read_MD_energy(mdout))
		md_s, md_p, md_f, final_dump = read_MD_dump(dumpout, elements)
		types = md_s[0]
		md_s = md_s[-1]
		md_p = md_p[-1]
		md_f = md_f[-1]
		
		nnp_e.append(md_e)
		nnp_f.append(md_f.flatten())

		symbols = []
		for e in range(len(types)):
			symbols.append(types[e])

		name2 = nnpDir.split('/')[-1]
		subprocess.call('mv min.dump '+'min.dump_%s' % name2,shell=True)
		min_dump.append('min.dump_%s' % name2)

	#Provide Energy and Forces for each structure from a single or multiple nnp
	return nnp_e, np.array(nnp_f), min_dump


def reax_n2p2_single_point(nnps, atoms, commands, cores, reax, reax_flag, elements=[]):

	base_path = os.getcwd()
	n2p2_path = commands['n2p2']
	lmp_path = commands['lmp']
	atomsk_path = commands['atomsk']

	rmse_info = []
	#For loop for structures
	#for i in range(len(images)):
	#atoms = images[i]

	if len(elements) == 0:
		elements= atoms.get_chemical_symbols()
		counter = collections.Counter(elements)

	element_short = list(dict.fromkeys(elements))
	element_ordered = list(sorted(element_short,key=lambda e:atomic_numbers[e]))

	chem = atoms.get_chemical_symbols()
	elements = list(dict.fromkeys(chem))
	if len(elements) == 0:
		elements_o = list(sorted(elements, key=lambda e: atomic_numbers[e]))
		atom_types = {}
		for el, j in zip(elements_o, range(len(elements_o))):
			atom_types[el]=elements.index(el) + 1
	else:
		elements_o = list(sorted(element_ordered, key=lambda e: atomic_numbers[e]))
		atom_types = {}
		for el, j in zip(elements_o, range(len(elements_o))):
			atom_types[el]=element_ordered.index(el) + 1

	lmp_filename = 'str.data'
	write_lammps_data(filename=lmp_filename,
					  atoms = atoms,
					  atom_types=atom_types,
					  units='real')

	#For loop for nnp for each structure
	nnp_e = []
	nnp_f = []
	reax_bo = []
	reax_species = []
	min_dump = []
	pair_rdf = []
	for nnpDir in nnps:
		write_reax_input(filename='nnp_input.in', nnpDir=nnpDir, elements=element_ordered, reax_flag=reax_flag, reax_ref=reax, nnp_cutoff=5.0001)
		lammps_command = "mpiexec -np %d " % int(cores) +  lmp_path + " -var filename " + lmp_filename + " -in nnp_input.in > out"
		call(lammps_command, shell=True)
		mdout = 'out'
		dumpout = 'min.dump'
		bofile = 'bonds.reaxc'
		species_out = 'species.out'
		with open(species_out,'r') as species:
			spc_contents = species.readlines()
		reax_species.append(spc_contents)
	
		md_e = float(read_MD_energy(mdout))
		md_s, md_p, md_f, final_dump = read_MD_dump(dumpout, elements)
		types = md_s[0]
		md_s = md_s[-1]
		md_p = md_p[-1]
		md_f = md_f[-1]
		
		nnp_e.append(md_e)
		nnp_f.append(md_f.flatten())

		symbols = []
		for e in range(len(types)):
			symbols.append(types[e])

		name2 = nnpDir.split('/')[-1]
		subprocess.call('mv min.dump '+'min.dump_%s' % name2,shell=True)
		subprocess.call('mv pair.rdf '+'pair.rdf_%s' % name2,shell=True)
		min_dump.append('min.dump_%s' % name2)
		pair_rdf.append('pair.rdf_%s' % name2)

		bo = read_BondOrder_new(bofile)
		reax_bo.append(bo)


	#Provide Energy and Forces for each structure from a single or multiple nnp
	return nnp_e, np.array(nnp_f), reax_species, reax_bo, min_dump, pair_rdf







