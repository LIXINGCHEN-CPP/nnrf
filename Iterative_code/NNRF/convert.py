#!/bin/env python
from ase.io import read, write
from ase.db import connect
from ase import Atoms, Atom
from ase.data import atomic_numbers, atomic_masses
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.visualize import view
#from ase.calculators.lammpslib import write_lammps_data
from .lammpslib import write_lammps_data
import sys, os
import numpy as np

def convert_pos2data(poscar_filename,
					 data_filename,
					 Triclinic=False,
					 elements_set=[]):

	atoms = read(poscar_filename)
	if len(elements_set) == 0:
		symbols_set = list(set(atoms.get_chemical_symbols()))
		elements_set = sorted(symbols_set, key=lambda x:atomic_numbers[x])

	atom_types = {}
	for el, i in zip(elements_set, range(len(elements_set))):
		atom_types[el] = i+1
	
	write_lammps_data(filename=data_filename, atoms=atoms, atom_types=atom_types, units='real')
	if not Triclinic:
		data = open(data_filename,"r")
		lines = data.readlines()

		new_data = open(data_filename,"w")
		for line in lines:
			if "xy xz yz" not in line:
				new_data.write(line)

	return

def convert_xyz2data(xyz_filename,
					 data_filename,
					 ):

	xyz_file = open(xyz_filename,'r')
	xyz_lines = xyz_file.readlines()
	images = []

	for i in range(len(xyz_lines)):
		line = xyz_lines[i]
		line_s = line.split()
		if len(line_s) == 1:
			Natoms = int(line_s[0])
			single_image = xyz_lines[i:i+Natoms+3]
			single_info = single_image[1].split()
			config_type = single_info[0]
			energy = float(single_info[1].split("=")[1]) # eV
			
			for j in range(len(single_info)):
				if "virial" in single_info[j]:
					virial_start = j
				if "Lattice" in single_info[j]:
					Lattice_start = j

			virial = single_info[virial_start:Lattice_start]
			virial[0] = virial[0].split('"')[1]
			virial[-1] = virial[-1].split('"')[0]
			virial_arr = np.array(virial).astype(float) #xx, yy, zz, yz, xz, xy: eV/Ang3

			lattice = single_info[Lattice_start:Lattice_start+9]
			lattice[0] = lattice[0].split('"')[1]
			lattice[-1] = lattice[-1].split('"')[0]
			lat = np.array([lattice[0:3],lattice[3:6],lattice[6:]]).astype(float)
			cell = lat
			
			
			atoms_list = np.array([ai.split() for ai in single_image[2:Natoms+2]])
			symbols = atoms_list[:,0]
			pos   = atoms_list[:,1:4].astype(float) #Ang
			forces = atoms_list[:,4:7].astype(float) #eV/Ang
			atoms = Atoms(symbols,positions=pos,cell=cell,pbc=[1,1,1])
			atoms.set_calculator(SPC(atoms, energy=energy,forces=forces,stress=virial))
			images.append(atoms)

	data_file_str = ""

	for i in range(len(images)):
		atoms = images[i]
		singledata = ""
		energy = atoms.get_potential_energy()
		positions = atoms.get_positions()
		symbols= atoms.get_chemical_symbols()
		forces = atoms.get_forces()
		cell = atoms.get_cell()
		
		singledata += "begin\n"
		singledata += "comment\n"
		singledata += "lattice %20.10f %20.10f %20.10f\n" % (cell[0][0],cell[0][1],cell[0][2])
		singledata += "lattice %20.10f %20.10f %20.10f\n" % (cell[1][0],cell[1][1],cell[1][2])
		singledata += "lattice %20.10f %20.10f %20.10f\n" % (cell[2][0],cell[2][1],cell[2][2])

		for ai in range(len(symbols)):
			e = symbols[ai]
			p = positions[ai]
			f = forces[ai]
			p_string = ["%20.10f" % pp for pp in p]
			f_string = ["%20.10f" % ff for ff in f]
			p_s = "".join(p_string)
			f_s = "".join(f_string)
			singledata += "atom %s %s %20.10f %20.10f %s\n" %(p_s, e, 0.0, 0.0, f_s)

		singledata += "energy %20.10f\n" % energy
		singledata += "charge 0\n"
		singledata += "end\n"
		data_file_str += singledata

	data_file = open(data_filename,'w')
	data_file.write(data_file_str)
	data_file.close()

	return data_filename

def convert_asedb2data(db_filename,
					   data_filename,
		             ):

	db = connect(db_filename)
	images = list(db.select())

	data_file_str = ""

	for i in range(len(images)):
		image = images[i]
		atoms = Atoms(image.symbols,
					  positions=image.positions,
					  cell=image.cell,
					  pbc=[1,1,1])
		atoms.set_calculator(SPC(atoms, energy=image.energy,forces=image.forces,stress=image.stress))
		singledata = ""
		energy = atoms.get_potential_energy()
		positions = atoms.get_positions()
		symbols= atoms.get_chemical_symbols()
		forces = atoms.get_forces()
		cell = atoms.get_cell()
		
		singledata += "begin\n"
		singledata += "comment\n"
		singledata += "lattice %20.10f %20.10f %20.10f\n" % (cell[0][0],cell[0][1],cell[0][2])
		singledata += "lattice %20.10f %20.10f %20.10f\n" % (cell[1][0],cell[1][1],cell[1][2])
		singledata += "lattice %20.10f %20.10f %20.10f\n" % (cell[2][0],cell[2][1],cell[2][2])

		for ai in range(len(symbols)):
			e = symbols[ai]
			p = positions[ai]
			f = forces[ai]
			p_string = ["%20.10f" % pp for pp in p]
			f_string = ["%20.10f" % ff for ff in f]
			p_s = "".join(p_string)
			f_s = "".join(f_string)
			singledata += "atom %s %s %20.10f %20.10f %s\n" %(p_s, e, 0.0, 0.0, f_s)

		singledata += "energy %20.10f\n" % energy
		singledata += "charge 0\n"
		singledata += "end\n"
		data_file_str += singledata

	data_file = open(data_filename,'w')
	data_file.write(data_file_str)
	data_file.close()

	return data_filename



def convert_dump_to_xyz_losing(dump_filename,
		                xyz_filename, cutoff_vol=400):

	xyz = open(xyz_filename,'w')
	with open(dump_filename,'r') as dump:
		all_lines = dump.readlines()
	filelength = len(all_lines)
	Natoms = int(all_lines[3])
	Natoms_pre = int(all_lines[3])

	Nimages = 0
	start = 0
	end = Natoms + 9
	single_timestep = all_lines[start:end]
	flag = True
	while flag:
		end = start + Natoms + 9 #1353
		if end >= filelength: 
			flag = False
		single_timestep = all_lines[start:end]
		Natoms = int(single_timestep[3])
		#print(single_timestep[1],Natoms)
		if Natoms != Natoms_pre:
			break

		#print(start, end, filelength)
		cell = single_timestep[5:8]
		cell2 = [each.split() for each in cell]
		if len(cell2[0]) == 3:
			xy = float(cell2[0][2])
			xz = float(cell2[1][2])
			yz = float(cell2[2][2])
		else:
			xy = 0
			xz = 0
			yz = 0
		xlo_bound = float(cell2[0][0]) - min(0.0, xy, xz, xy+xz)
		xhi_bound = float(cell2[0][1]) - max(0.0, xy, xz, xy+xz)
		ylo_bound = float(cell2[1][0]) - min(0.0, yz)
		yhi_bound = float(cell2[1][1]) - max(0.0, yz)
		zlo_bound = float(cell2[2][0])
		zhi_bound = float(cell2[2][1])
		x = xhi_bound - xlo_bound
		y = yhi_bound - ylo_bound
		z = zhi_bound - zlo_bound

		#print(xlo_bound, xhi_bound, ylo_bound, yhi_bound, zlo_bound, zhi_bound, xy, xz, yz)
		if start == 0:
			starting_vol = x*y*z

		volume = x*y*z
		volume_percent = 100 * (volume - starting_vol) / starting_vol

		if volume_percent >= cutoff_vol:
			break
		else:
			Nimages += 1

		cell_orth = np.array([[x,0,0],[xy,y,0],[xz,yz,z]])
		cell_orth = cell_orth.flatten()
		cell_orth = cell_orth.astype(float)
		cell_string = " ".join([str(i) for i in cell_orth])
		tags = single_timestep[8]
		atoms = single_timestep[9:]
		#cell_orth = [float(i.split()[1]) for i in cell]
		names = tags.split()[2:]
		#get id index from names
		id_index = names.index('id')
		atoms_c = np.array(sorted([atom.split() for atom in atoms],key=lambda x: int(x[id_index])))
		#type_index = names.index('element')
		#type_index = names.index('type')
		element_index = names.index('element')
		elements = sorted(list(set(atoms_c[:,element_index])), key=lambda x:atomic_numbers[x])
		chemical_symbols = atoms_c[:,element_index]
		#chemical_symbols = [elements[int(a)-1] for a in chemical_symbols ]
		formula = ''.join(chemical_symbols)
		x_index = names.index('x')
		y_index = names.index('y')
		z_index = names.index('z')
		symbols = np.array( [[element] for element in chemical_symbols] )
		p_x = np.array( [[float(atom)] for atom in atoms_c[:,x_index]] )
		p_y = np.array( [[float(atom)] for atom in atoms_c[:,y_index]] )
		p_z = np.array( [[float(atom)] for atom in atoms_c[:,z_index]] )

		p_x_nan = np.isnan(np.sum(p_x))
		p_y_nan = np.isnan(np.sum(p_y))
		p_z_nan = np.isnan(np.sum(p_z))
		if p_x_nan or p_y_nan or p_z_nan:
			break
		

		p_x = p_x - xlo_bound
		p_y = p_y - ylo_bound
		p_z = p_z - zlo_bound
		atoms_xyz = np.hstack((symbols, p_x,p_y,p_z))
		xyz.write("%d\n" % len(atoms_c))
		#cell_string = "%f 0.0 0.0 0.0 %f 0.0 0.0 0.0 %f" % (cell_orth[0], cell_orth[1], cell_orth[2])
		xyz.write("Lattice='"+cell_string+"' Properties=species:S:1:pos:R:3 Timestep:=T pbc="'F F F'" 1229=T Atoms.=T\n")
		for i in range(len(atoms_xyz)):
			xyz.write(' '.join([str(a) for a in atoms_xyz[i]]) + '\n')
		start += len(single_timestep)
		Natoms_pre = Natoms
	xyz.close()

	return Nimages

def convert_dump_to_xyz_complete(dump_filename,
		                xyz_filename, cutoff_vol=10000):

	xyz = open(xyz_filename,'w')
	with open(dump_filename,'r') as dump:
		all_lines = dump.readlines()
	filelength = len(all_lines)
	Natoms = int(all_lines[3])
	Natoms_pre = int(all_lines[3])

	Nimages = 0
	start = 0
	end = Natoms + 9
	single_timestep = all_lines[start:end]
	flag = True
	while flag:
		end = start + Natoms + 9 
		if end >= filelength: 
			flag = False
		single_timestep = all_lines[start:end]
		Natoms = int(single_timestep[3])
		#if Natoms != Natoms_pre:
		#	break

		#print(start, end, filelength)
		cell = single_timestep[5:8]
		cell2 = [each.split() for each in cell]
		if len(cell2[0]) == 3:
			xy = float(cell2[0][2])
			xz = float(cell2[1][2])
			yz = float(cell2[2][2])
		else:
			xy = 0
			xz = 0
			yz = 0
		xlo_bound = float(cell2[0][0]) - min(0.0, xy, xz, xy+xz)
		xhi_bound = float(cell2[0][1]) - max(0.0, xy, xz, xy+xz)
		ylo_bound = float(cell2[1][0]) - min(0.0, yz)
		yhi_bound = float(cell2[1][1]) - max(0.0, yz)
		zlo_bound = float(cell2[2][0])
		zhi_bound = float(cell2[2][1])
		x = xhi_bound - xlo_bound
		y = yhi_bound - ylo_bound
		z = zhi_bound - zlo_bound

		if start == 0:
			starting_vol = x*y*z

		volume = x*y*z
		volume_percent = 100 * (volume - starting_vol) / starting_vol

		if volume_percent >= cutoff_vol:
			break
		else:
			Nimages += 1

		cell_orth = np.array([[x,0,0],[xy,y,0],[xz,yz,z]])
		cell_orth = cell_orth.flatten()
		cell_orth = cell_orth.astype(float)
		cell_string = " ".join([str(i) for i in cell_orth])
		tags = single_timestep[8]
		atoms = single_timestep[9:]
		#cell_orth = [float(i.split()[1]) for i in cell]
		names = tags.split()[2:]
		#get id index from names
		id_index = names.index('id')
		atoms_c = np.array(sorted([atom.split() for atom in atoms],key=lambda x: int(x[id_index])))
		#type_index = names.index('element')
		type_index = names.index('type')
		element_index = names.index('element')
		elements = sorted(list(set(atoms_c[:,element_index])), key=lambda x:atomic_numbers[x])
		chemical_symbols = atoms_c[:,element_index]
		#chemical_symbols = atoms_c[:,type_index]
		#chemical_symbols = [elements[int(a)-1] for a in chemical_symbols ]
		formula = ''.join(chemical_symbols)
		x_index = names.index('x')
		y_index = names.index('y')
		z_index = names.index('z')
		symbols = np.array( [[element] for element in chemical_symbols] )
		p_x = np.array( [[float(atom)] for atom in atoms_c[:,x_index]] )
		p_y = np.array( [[float(atom)] for atom in atoms_c[:,y_index]] )
		p_z = np.array( [[float(atom)] for atom in atoms_c[:,z_index]] )

		p_x = p_x - xlo_bound
		p_y = p_y - ylo_bound
		p_z = p_z - zlo_bound
		atoms_xyz = np.hstack((symbols, p_x,p_y,p_z))
		xyz.write("%d\n" % len(atoms_c))
		#cell_string = "%f 0.0 0.0 0.0 %f 0.0 0.0 0.0 %f" % (cell_orth[0], cell_orth[1], cell_orth[2])
		xyz.write("Lattice='"+cell_string+"' Properties=species:S:1:pos:R:3 Timestep:=T pbc="'F F F'" 1229=T Atoms.=T\n")
		for i in range(len(atoms_xyz)):
			xyz.write(' '.join([str(a) for a in atoms_xyz[i]]) + '\n')
		start += len(single_timestep)
		Natoms_pre = Natoms
		if filelength > end+3:
			Natoms = int(all_lines[end+3])


	xyz.close()

	return Nimages

