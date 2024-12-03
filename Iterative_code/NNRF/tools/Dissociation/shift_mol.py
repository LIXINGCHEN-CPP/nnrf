from ase.io import read, write
from ase import Atoms, Atom
from ase.visualize import view
import copy, os
import numpy as np


mols_xyz = read('total.xyz',index=':')
mols_bo_sync = []
index_sync = []

####Collect BO table
mols_bo  = open('total_bo.reaxc','r')
bo_lines = mols_bo.readlines()
for i in range(len(bo_lines)):
	line = bo_lines[i]
	line_s = line.split()
	if len(line_s) > 1 and line_s[1] == 'Timestep':
		Natoms = int(bo_lines[i+2].split()[-1])
		start = i
		end = i+Natoms+8
		mol_bo =[[j]+l.split() for j, l in enumerate(bo_lines[start+7:end-1])]
		mols_bo_sync.append(mol_bo)

mols_index = open('total.xyz','r')
index_lines = mols_index.readlines()
for i in range(len(index_lines)):
	line = index_lines[i]
	line_s = line.split()
	if len(line_s) == 1 and line_s[0].isdigit():
		Natoms = int(line_s[0])
		start = i
		end = i+Natoms+2
		atoms_chunk = np.array([t.split() for t in index_lines[start+2:end]])
		index_sync.append(atoms_chunk[:,4])

test = []
for i in range(len(mols_xyz)):
	if len(mols_xyz[i]) < 30 and len(mols_xyz[i]) > 3:
		molecule = mols_xyz[i]	
		bo_table = mols_bo_sync[i]
		info = []
		for a in range(len(bo_table)):
			bo = bo_table[a]
			indexx = list(index_sync[i])
			seq = list(range(len(indexx)))
			n_neighbor = int(bo[3])
			n_index = bo[4:4+n_neighbor]
			atom_i = indexx.index(bo[1])
			nn_i = [indexx.index(n_) for n_ in n_index]
			
			atom = molecule[atom_i]
			n_atom = [molecule[c] for c in nn_i]
			info.append([atom,n_atom])

		Count_bond = 0
		diss_bond = []
		for bc1 in range(len(info)):
			nn_info = info[bc1][1]
			for bc2 in range(len(nn_info)):
				if 'N' == info[bc1][0].symbol and 'N' == nn_info[bc2].symbol:
					a1_i = info[bc1][0].index
					a2_i = nn_info[bc2].index
					if [a2_i,a1_i] not in diss_bond:
						Count_bond += 1
						diss_bond.append([info[bc1][0].index,nn_info[bc2].index])

		cell = molecule.get_cell()
		pos = molecule.get_positions()
		ave_pos_x = np.median(pos[:,0]) + 1
		ave_pos_y = np.median(pos[:,1]) + 1
		ave_pos_z = np.median(pos[:,2]) + 1
		shift_x = 0.5 * cell[0][0] - ave_pos_x
		shift_y = 0.5 * cell[1][1] - ave_pos_y
		shift_z = 0.5 * cell[2][2] - ave_pos_z

		pos[:,0] += shift_x
		pos[:,1] += shift_y
		pos[:,2] += shift_z

		pos[:,0] = [x-cell[0][0] if x > cell[0][0] else x for x in pos[:,0]] 
		pos[:,0] = [x+cell[0][0] if x < 0 else x for x in pos[:,0]] 

		pos[:,1] = [x-cell[1][1] if x > cell[1][1] else x for x in pos[:,1]] 
		pos[:,1] = [x+cell[1][1] if x < 0 else x for x in pos[:,1]] 

		pos[:,2] = [x-cell[2][2] if x > cell[2][2] else x for x in pos[:,2]] 
		pos[:,2] = [x+cell[2][2] if x < 0 else x for x in pos[:,2]] 

		molecule.set_positions(pos)	

		if Count_bond > 0:
			for l in range(Count_bond):
				new_atoms = copy.deepcopy(molecule)
				pos2 = new_atoms.get_positions()
				one_a = diss_bond[l][0]
				two_a = diss_bond[l][1]
				one_ap = new_atoms[one_a].position
				two_ap = new_atoms[two_a].position
				pos2[:,0] -= one_ap[0]
				pos2[:,1] -= one_ap[1]
				pos2[:,2] -= one_ap[2]
				new_atoms.set_positions(pos2)
				new_atoms.rotate(two_ap, (1,0,0))
				write('shifted_%d_%d.xyz' % (i,l),new_atoms)

			

#write('shifted.xyz',test)
		
