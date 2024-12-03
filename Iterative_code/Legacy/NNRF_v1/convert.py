#!/bin/env python
from ase.io import read, write
from ase import Atoms, Atom
from ase.data import atomic_numbers, atomic_masses
from ase.visualize import view
import sys, os
import numpy as np

def convert_db_to_data():
	return




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

