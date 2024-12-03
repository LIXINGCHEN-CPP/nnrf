#!/bin/env python
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.data import atomic_numbers, atomic_masses
from ase import Atoms, Atom
from ase.db import connect
from ase.visualize import view
from ase.units import eV, kcal, mol
from ase.neighborlist import NeighborList
from ase.io import read, write
from .lammpslib import write_lammps_data
from .tracking import job_submit_file
import sys, os, glob, re, json, time, subprocess, pybel, scipy, math, copy
import collections
import pandas as pd
import numpy as np
from shutil import copyfile
from decimal import Decimal
from numpy import *

def atomsk(basename, atomsk_path, xyz_filename):
	atomsk_command = atomsk_path +  " " + xyz_filename + " lmp -properties charge.txt"
	subprocess.call(atomsk_command, shell=True)
	return

def lammps(lmp_filename, lmp_path, ncores):
	lammps_command ="mpiexec -np "+str(ncores)+" " + lmp_path + " -var filename " + lmp_filename + " -in base.in > lmp.out"
	subprocess.call(lammps_command, shell=True)
	return

def chargefile(filename, elements):
	with open(filename, 'w') as c1:
		c1.write('charge\n')
		elements = list(set(elements))
		for element in elements:
			c1.write(element + ' 0.0\n')


def input_file2(filename, elements, ffield, ffoption=''):
	elements0 = sorted(elements,key=lambda x: atomic_numbers[x])
	e_string = " ".join(elements0)
	with open('lmp_control','w') as c:
		string0 = "simulation_name		ReaxFF ! output files will carry this name + their specific ext\n"
		string0 += "tabulate_long_range	10000 ! denotes the granularity of long range tabulation, 0 means no tabulation\n"
		string0 += "energy_update_freq 	1\n"
		string0 += "nbrhood_cutoff		5.0  ! near neighbors cutoff for bond calculations in A\n"
		string0 += "hbond_cutoff		6.0  ! cutoff distance for hydrogen bond interactions 6.0 default\n"
		string0 += "bond_graph_cutoff	0.3  ! bond strength cutoff for bond graphs\n"
		string0 += "thb_cutoff		0.001 ! cutoff value for three body interactions\n"
		string0 += "write_freq		100000    ! write trajectory after so many steps\n"
		string0 += "traj_title		RDX_ReaxFF ! (no white spaces)\n"
		string0 += "atom_info		0    ! 0: no atom info, 1: print basic atom info in the trajectory file\n"
		string0 += "atom_forces		0    ! 0: basic atom format, 1: print force on each atom in the trajectory file\n"
		string0 += "atom_velocities		0    ! 0: basic atom format, 1: print the velocity of each atom in the trajectory file\n"
		string0 += "bond_info		0    ! 0: do not print bonds, 1: print bonds in the trajectory file\n"
		string0 += "angle_info		0    ! 0: do not print angles, 1: print angles in the trajectory file\n"
		c.write(string0)

	with open(filename,'w') as ff:
		string = "units           real\n"
		string += "boundary        p p p\n"
		string += "atom_style      charge\n"
		string += "neighbor        2.0 nsq\n"
		string += "neigh_modify    delay 2\n"
		string += "box             tilt large\n"
		string += "read_data       ${filename}.reax\n\n"

		string += "#    Potential Parameters\n"
		string += "pair_style      reax/c lmp_control safezone 4.0 mincap 400 "+ ffoption +"\n"
		string += "pair_coeff      * * "+ffield +" " + e_string + "\n"
		string += "compute reax all pair reax/c\n"
		string += "fix             10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n"
		string += "variable eb equal c_reax[1]\n"
		string += "variable ea equal c_reax[2]\n"
		string += "variable elp equal c_reax[3]\n"
		string += "variable emol equal c_reax[4]\n"
		string += "variable ev equal c_reax[5]\n"
		string += "variable epen equal c_reax[6]\n"
		string += "variable ecoa equal c_reax[7]\n"
		string += "variable ehb equal c_reax[8]\n"
		string += "variable et equal c_reax[9]\n"
		string += "variable eco equal c_reax[10]\n"
		string += "variable ew equal c_reax[11]\n"
		string += "variable ep equal c_reax[12]\n"
		string += "variable efi equal c_reax[13]\n"
		string += "variable eqeq equal c_reax[14]\n"
		string += "thermo_style    custom step v_eb v_ea v_elp v_emol v_ev v_epen v_ecoa v_ehb v_et v_eco v_ew v_ep v_efi v_eqeq pe pxx pyy pzz pxy pxz pyz\n"
		string += "thermo   1\n"

		string += "fix             rbo all reax/c/bonds 2 bonds.reaxc_${filename}\n"
		string += "dump            min all custom 2 min.dump_${filename} id type x y z q fx fy fz\n"
		string += "dump            d1 all custom 1 force.dump fx fy fz\n"
		string += "dump_modify     d1 sort id\n"
		string += "min_style       cg\n"
		string += "minimize        1e-8 1e-8  0  0\n"
		string += "run             1\n"
		ff.write(string)

	return

# Extract atom types from datafile and push into an array
def extract_atom_types(datafile):
    ifile = open(datafile, 'r')
    var_dummy = 0 # dummy variable
    # Loops through entire datafile
    for line in ifile:
        items = str.split(line)
        # Extracts number of atom types
        if(len(items) == 3 and items[1] == 'atom' and items[2] == 'types'):
            array_types = np.zeros(int(items[0]))
        # Increments dummy variable
        elif('Masses' in line):
            var_dummy += 1
        # Pushes mass to array
        elif(var_dummy >= 1 and len(items) == 2):
            array_types[var_dummy-1] = items[1]
            var_dummy += 1
        # Breaks out of loop
        elif('Atoms' in line):
            break
    return array_types

			
# Groups clusters based on bond order table
# Syntax [[[(..),(..),(..)]]] Layer 1: (atom_id, atom_type), Layer 2: [clusters], Layer 3: [Timeframe], Layer 4: [Entire file]
def read_BOTable(BOTable):
    ifile = open(BOTable, 'r')
    list_timesteps = [] # list of timestep numbers (length = number of bond order frames)
    single_timeframe = [] # list of all rows, includes bonded atom_ids and atom_type of index 0
    cluster_atom_ids = [] # list of clusters of atom_ids
    cluster_tuples = [] # list of cluster tuples (atom_id , atom_type)
    single_frame_clusters = [] # list of cluster tuples for a single timeframe
    clusters_file = [] # list of cluster tuples for entire file
    var_dummy = 0 # dummy variable
    num_clusters = 0
    for line in ifile:
        # Extracts timestep number
        if("# Timestep" in line):
            list_timesteps.append(eval(str.split(line)[2]))
        elif("# id type" in line):
            var_dummy += 1
        elif(var_dummy== 1):
            items = str.split(line)
            # Determining clusters after reading single timeframe
            if(items[0] == "#"):
                var_dummy = 0
                timeframe_original = np.copy(single_timeframe)
                cluster_atom_ids.append(single_timeframe[0][:-1])
                len_timeframe = len(single_timeframe) # Length of timeframe
                cluster_index = 0 # Index of initial cluster
                # Looping over all atom_ids in single cluster
                while(cluster_index < len(cluster_atom_ids[num_clusters])):
                    current_atom_id = cluster_atom_ids[num_clusters][cluster_index] # atom_id of interest in cluster list
                    # Looping over remaining rows in timeframe
                    for row in single_timeframe:
                        if(row[0] == current_atom_id):
                            # Appends grouped atom_ids to cluster
                            updated_cluster = remove_duplicates(cluster_atom_ids[num_clusters]+row[:-1])
                            single_timeframe.remove(row)
                            break
                    cluster_atom_ids[num_clusters] = updated_cluster
                    len_timeframe = len(single_timeframe)
                    cluster_index += 1
                    # If all atom_ids in cluster, move to next cluster
                    if((cluster_atom_ids[num_clusters].index(current_atom_id) == len(cluster_atom_ids[num_clusters])-1) and (len(single_timeframe) != 0)):
                        cluster_atom_ids.append(single_timeframe[0][:-1])
                        num_clusters += 1
                        cluster_index = 0 # Index of next cluster
                # Loop through all clusters and make tuples of structure (atom_id,atom_type)
                for cluster in cluster_atom_ids:
                    for atom_id in cluster:
                        for row_original in timeframe_original:
                            if(row_original[0]  == atom_id):
                                cluster_tuples.append((atom_id,row_original[-1]))
                    single_frame_clusters.append(cluster_tuples)
                    cluster_tuples = []
                clusters_file.append(single_frame_clusters)
                single_frame_clusters = []
                cluster_atom_ids = []
                single_timeframe = []
                var_dummy = 0
                num_clusters = 0
                continue
            else:
                num_bonds = int(items[2])
                templist = [items[0]]
                templist.extend(items[3:3+num_bonds])
                templist.extend(items[1])
                single_timeframe.append(templist)
    ifile.close()
    return clusters_file, list_timesteps

# Removes duplicate indices in list without ordering
def remove_duplicates(li):
    my_set = set()
    no_dups = []
    for element in li:
        if element not in my_set:
            no_dups.append(element)
            my_set.add(element)
    return no_dups

# Convert atom_type masses to element type
def element_types(atom_types):
    mass = float(atom_types)
    element_str = ''
    if(0.9 <= mass and mass <= 1.1):
        element_str = 'H'
    elif(11.9 <= mass and mass <= 12.1):
        element_str = 'C'
    elif(13.9 <= mass and mass <= 14.1):
        element_str = 'N'
    elif(15.9 <= mass and mass <= 16.1):
        element_str = 'O'
    elif(65.2 <= mass and mass <= 65.5):
        element_str = 'Zn'
    return element_str

def cluster_list_to_molecule_str(atom_types, clusters_file):
    element_types_array = [] # array of element_types
    cluster_atom_ids = []
    cluster_element_types = []
    individual_cluster_frame = []
    updated_clusters_file = []
    # Determines element type from atom type
    for index in range(len(atom_types)):
        element_types_array.append(str(element_types(atom_types[index])))
    #print(element_types_array)
    elm_size = len(element_types_array)
    updated_element_types = np.zeros(elm_size) # Updated list for atom_type which counts how many of each element is listed
    corrected_element_types = np.zeros(elm_size)
    # Builds tuples of ([atom_ids], cluster_str)
    for single_cluster_frame in clusters_file:
        for cluster in range(len(single_cluster_frame)):
            for atom_id_index in range(len(single_cluster_frame[cluster])):
                corrected_element_types[int(single_cluster_frame[cluster][atom_id_index][1])-1] += 1
                cluster_atom_ids.append((single_cluster_frame[cluster][atom_id_index][0]))
            cluster_str = ''
            for element_id_index in range(elm_size):
                if(int(corrected_element_types[element_id_index]) == 0):
                    continue
                elif(int(corrected_element_types[element_id_index]) != 1):
                    cluster_str += element_types_array[element_id_index]+str(int(corrected_element_types[element_id_index]))
                else:
                    cluster_str += element_types_array[element_id_index]
            cluster_element_types.append(cluster_str)
            individual_cluster_frame.append((sorted(cluster_atom_ids),cluster_element_types[-1]))
            cluster_atom_ids = []
            corrected_element_types = np.zeros(elm_size)
            cluster_element_types = []
        updated_clusters_file.append(individual_cluster_frame)
        individual_cluster_frame = []
    return updated_clusters_file

# Determines timeframes where cluster_id exists, two files: 1) timestamp, 2) atom ids of individual species
def species_timeframes(clusters_file,species_name):
    times = []
    mol_ids = []
    mol_ids_unsort = []
    timestep = 1000
    timeframe_num = 0
    species_interest = str(species_name)
    for single_cluster_frame in clusters_file:
        time = []
        mol_id = []
        for cluster in single_cluster_frame:
            if(cluster[-1] == species_interest):
                for k in range(len(cluster[0])):
                    time.append(int(timeframe_num*timestep))
                    mol_id.append(cluster[0][k])
                mol_ids_unsort.append((cluster[:-1],int(timeframe_num*timestep)))
        mol_id = np.array(mol_id,dtype=int)
        mol_id = sorted(mol_id)
        mol_id = list(mol_id)
        times.extend(time)
        mol_ids.extend(mol_id)
        timeframe_num += 1
    tp = np.array(times)
    fp = np.array(mol_ids)
    #molecules_file = np.vstack((tp, fp)).T
    #return (molecules_file)
    return tp,fp,mol_ids_unsort

def print_lines(time,molid):
    header = 9
    size = 3132
    multi = 0
    lines = []
    for k in range(len(time)):
        multi = int(time[k])/1000 #step multiplier
        if(multi == 0):
            lines.append(int(molid[k])+header)
        else:
            lines.append(int(molid[k])+((size+header)*multi)+header)
    molecules_file = np.vstack((time, molid, lines)).T
    return molecules_file



#PS
def extract_coordinate(dumpfile,bofile,cluster_str, types, t_info, exclude=[]):
	with open(dumpfile, 'r') as f:
		dumpfile_str = f.readlines()
	with open(bofile, 'r') as f_bo:
		bofile_str = f_bo.readlines()

	bo_list = []
	bo_header = []
	for b in range(len(bofile_str)):
		line = bofile_str[b].split()

		if len(line)>2 and line[1] == 'Timestep':
			Natoms = int(bofile_str[b+2].split()[-1])
			start = b+7
			end = b+7+Natoms
			bo_list.append(bofile_str[start:end])
			bo_header.append(bofile_str[b:start])

	#each timestep
	k = 0
	species0 = []
	species_acc = []
	filelist = []
	bo_filelist = []
	f1 = open("total.xyz","w")
	f1_bo = open("total_bo.reaxc","w")
	str0 = ""
	str0_bo = ""
	timestep_time = 0
	for i in range(len(cluster_str)):
		f2 = open("mol_%s.xyz" % t_info[i],"w")
		f2_bo = open("mol_bo_%s.reaxc" % t_info[i],"w")
		filelist.append("mol_%s.xyz" % t_info[i])
		bo_filelist.append("mol_bo_%s.reaxc" % t_info[i])
		header0 = dumpfile_str[0+k:9+k]
		natoms = int(header0[3].split()[0])
		positions = dumpfile_str[9+k:9+k+natoms]
		positions = [line.split() for line in positions]
		species1 = []
		pos_species1 = []
		natoms0 = []
		for j in range(len(cluster_str[i])):
			mol_index = cluster_str[i][j][0]
			mol_chem  = cluster_str[i][j][1]
					 
			natoms0.append(len(mol_index))
			species1.append(mol_chem)
			#collect xyz coordinate
			xyz = []
			for index in mol_index:
				pos_line = list(filter(lambda x: x[0] == index, positions))
				xyz.append(pos_line[0])

			if not xyz[0][1].isdigit():
				for e in xyz:
					del e[1]
				xyz = sorted(xyz, key=lambda x: int(x[1]))
			else:
				xyz = sorted(xyz, key=lambda x: int(x[1]))
			pos_species1.append(xyz)

		species_types = list(set(species1))
		#write individual files based on species1, pos_species1 and header0
		
		bo_list_time = sorted([line.split() for line in bo_list[i]],key=lambda x: int(x[1]))
		bo_header_time = bo_header[i]
		timestep_time = int(bo_header_time[0].split(" ")[2])

		for j in range(len(species1)):
			header1 = header0[:]
			header1[3] = str(natoms0[j]) + "\n"
			cell = header1[5:8]
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

			cell_orth = np.array([[x,0,0],[xy,y,0],[xz,yz,z]])
			cell_orth = cell_orth.flatten()
			cell_orth = cell_orth.astype(float)
			cell_string = " ".join([str(i) for i in cell_orth])
			for l in range(len(pos_species1[j])):
				pos_species1[j][l][2] = str(float(pos_species1[j][l][2]) - float(xlo_bound))
				pos_species1[j][l][3] = str(float(pos_species1[j][l][3]) - float(ylo_bound))
				pos_species1[j][l][4] = str(float(pos_species1[j][l][4]) - float(zlo_bound))
			header1[5] = " ".join( [str(xlo_bound), str(xhi_bound), str(xy)] ) + "\n"
			header1[6] = " ".join( [str(ylo_bound), str(yhi_bound), str(xz)] ) + "\n"
			header1[7] = " ".join( [str(zlo_bound), str(zhi_bound), str(yz)]) + "\n"
			header = " ".join(header1) + " "
			pos2 = [" ".join(line) + "\n" for line in pos_species1[j]]
			dump_line = " ".join(pos2)
			dump_file = header + dump_line
			str1 = str(natoms0[j]) + "\n"
			str1 += "Lattice='"+cell_string+"' Properties=species:S:1:pos:R:3 Timestep:=T pbc="'F F F'" 1229=T Atoms.=T\n"
			track_index = []
			track_bo = []
			for l in range(len(pos_species1[j])):
				track_index.append(pos_species1[j][l][0])
				str1 += types[pos_species1[j][l][1]] + " " #atomic element
				if len(pos_species1[j]) - 1 == l:
					str1 += " ".join(pos_species1[j][l][2:5]) + " " + pos_species1[j][l][0] + " \n"
				else:
					str1 += " ".join(pos_species1[j][l][2:5]) + " " + pos_species1[j][l][0] + " \n"
			f2.write(str1)

			for bi in range(len(bo_list_time)):
				if bo_list_time[bi][0] in track_index:
					track_bo.append(bo_list_time[bi])
			bo_header_time[2] = '# Number of particles %d \n' % natoms0[j]
			str1_bo = "".join(bo_header_time)
			for bo in range(len(track_bo)):
				str1_bo += " "+" ".join(track_bo[bo]) +"\n"
			str1_bo += "#\n"
			f2_bo.write(str1_bo)

			####
			sp = re.findall('[A-Z][^A-Z]*', species1[j])
			sp_ = {}
			for single in sp:
				chm = ''
				num = 1
				for ll in range(len(single)):
					letter = single[ll]
					if not letter.isdigit():
						chm += letter
					elif letter.isdigit() and int(letter) !=0:
						num = int(single[ll:])
				sp_[chm] = num

			seq = ['C','H','N','O']
			chemical_sym = ''
			keys = sp_.keys()
			for ee in seq:
				if ee in keys:
					if sp_[ee] != 1:
						chemical_sym += ee + str(sp_[ee])
					elif sp_[ee] == 1:
						chemical_sym += ee
			####
			#print(chemical_sym)
			species_acc = list(set(species_acc))
			if chemical_sym not in exclude and chemical_sym not in species_acc:
				species_acc.append(chemical_sym)
				str0 += str(natoms0[j]) + "\n"
				str0 += "Lattice='"+cell_string+"' Properties=species:S:1:pos:R:3 Timestep:=T pbc="'F F F'" 1229=T Atoms.=T\n"

				track_index2 = []
				track_bo2 = []
				for l in range(len(pos_species1[j])):
					track_index2.append(pos_species1[j][l][0])
					str0 += types[pos_species1[j][l][1]] + " " #atomic element
					if len(pos_species1[j]) - 1 == l:
						str0 += " ".join(pos_species1[j][l][2:5]) + " " + pos_species1[j][l][0] + " \n"
					else:
						str0 += " ".join(pos_species1[j][l][2:5]) + " " + pos_species1[j][l][0] + " \n"

				for bi in range(len(bo_list_time)):
					if bo_list_time[bi][0] in track_index2:
						track_bo2.append(bo_list_time[bi])
				bo_header_time[0] = '# Timestep %d\n' % timestep_time
				str0_bo += "".join(bo_header_time) 
				for bo in range(len(track_bo2)):
					str0_bo +=" "+" ".join(track_bo2[bo]) + "\n"
				str0_bo += "#\n"

		k = natoms + k + 9
	f1.write(str0)
	f1_bo.write(str0_bo)
	return filelist

def examine_element_neighbor(BOfile, types, target_element, nn):
	
    bofile = open(BOfile, 'r').readlines()

    each_timestep = []
    timestep_n = []
    for i in range(len(bofile)):
        if len(bofile[i].split()) > 2:
            if "# Timestep " in bofile[i]:
                start = i
                Natoms = bofile[i+2].split()[4]
                end = i + int(Natoms) + 7
                each_timestep.append(bofile[start+7:end])
                timestep_n.append(bofile[i].split()[-1])

    f = open('count.txt','w')
    for i in range(len(each_timestep)):   
        chunk = each_timestep[i]
        timestep = timestep_n
        count = 0
        for j in range(len(chunk)):
            line = chunk[j].split()
            type1 = types[line[1]]
            if type1 == target_element:
                index = line[0]
                Nneighbors = line[2]
                if Nneighbors == nn:
                    count += 1
        f.write("%s %d\n" % (timestep[i], count))
    f.close()

    return

def atoi(text):
    if(text.isdigit() == True):
        return int(text)
    else:
        return text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

def dists(xrng,yrng,zrng,x2,x1,y2,y1,z2,z1): #Returns least distance between atoms or periodic images    d1=((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)**0.5
	d1=((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)**0.5
	d2=((x2+xrng-x1)**2+(y2-y1)**2+(z2-z1)**2)**0.5
	d3=((x2-xrng-x1)**2+(y2-y1)**2+(z2-z1)**2)**0.5
	d4=((x2-x1)**2+(y2+yrng-y1)**2+(z2-z1)**2)**0.5
	d5=((x2-x1)**2+(y2-yrng-y1)**2+(z2-z1)**2)**0.5
	d6=((x2-x1)**2+(y2-y1)**2+(z2+zrng-z1)**2)**0.5
	d7=((x2-x1)**2+(y2-y1)**2+(z2-zrng-z1)**2)**0.5
	d8=((x2+xrng-x1)**2+(y2+yrng-y1)**2+(z2-z1)**2)**0.5
	d9=((x2-xrng-x1)**2+(y2-yrng-y1)**2+(z2-z1)**2)**0.5
	d10=((x2-x1)**2+(y2+yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d11=((x2-x1)**2+(y2-yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	d12=((x2+xrng-x1)**2+(y2-y1)**2+(z2+zrng-z1)**2)**0.5
	d13=((x2-xrng-x1)**2+(y2-y1)**2+(z2-zrng-z1)**2)**0.5
	d14=((x2+xrng-x1)**2+(y2+yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d15=((x2-xrng-x1)**2+(y2-yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	d16=((x2+xrng-x1)**2+(y2-yrng-y1)**2+(z2-z1)**2)**0.5
	d17=((x2-xrng-x1)**2+(y2+yrng-y1)**2+(z2-z1)**2)**0.5
	d18=((x2-x1)**2+(y2+yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	d19=((x2-x1)**2+(y2-yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d20=((x2+xrng-x1)**2+(y2-y1)**2+(z2-zrng-z1)**2)**0.5
	d21=((x2-xrng-x1)**2+(y2-y1)**2+(z2+zrng-z1)**2)**0.5
	d22=((x2+xrng-x1)**2+(y2+yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	d23=((x2+xrng-x1)**2+(y2-yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d24=((x2-xrng-x1)**2+(y2+yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d25=((x2-xrng-x1)**2+(y2-yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d26=((x2+xrng-x1)**2+(y2-yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	d27=((x2-xrng-x1)**2+(y2+yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	return min(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27)

def Rcutoff_botable(trajectory, working_dir, interval, 
					rdict={'11':'1.30','22':'1.55','33':'1.60','44':'1.60',
						   '12':'1.20','21':'1.20','23':'1.65','32':'1.65',
						   '24':'1.65','42':'1.65','13':'1.30','31':'1.30', 
						   '14':'1.30','41':'1.30','34':'1.60','43':'1.60'}):
	os.chdir(working_dir)
	images = read(trajectory,index=':')
	natoms = len(images[0])

	x = np.zeros(natoms, float)
	y = np.zeros(natoms, float)
	z = np.zeros(natoms, float)
	typ = np.zeros(natoms, int)
	ids = np.zeros(natoms, int)
	elements = list(sorted(set(images[0].get_chemical_symbols()), key=lambda x: atomic_numbers[x]))
	types = [i+1 for i in range(len(elements))]
	
	ofile=open('bonds.reaxc','w')
	dfile=open('dump_reax.dump','w')
	for k in range(0,len(images),interval):
		atoms = images[k]
		chem = atoms.get_chemical_symbols()
		elements = sorted(list(set(chem)), key=lambda x: atomic_numbers[x])
		atom_types = {}
		for el, j in zip(elements, range(len(elements))):
			atom_types[el] = j+1
		if k == 0:
			write_lammps_data(filename='data.in', atoms=atoms, atom_types=atom_types, units='real')
		fudge_fact=1.0
		E = {}
		for e in range(len(elements)):
			E[elements[e]] = types[e]

		pos = images[k].get_positions()
		cell = images[k].get_cell_lengths_and_angles()

		lx = cell[0]
		xy = cell[1] * np.cos(np.deg2rad(cell[5]))
		xz = cell[2] * np.cos(np.deg2rad(cell[4]))
		ly = np.sqrt(cell[1]**2 - xy**2)
		yz = (cell[1]*cell[2]*np.cos(np.deg2rad(cell[3])) - xy*xz)/ly
		lz = np.sqrt(cell[2]**2 - xz**2 - yz**2)
		xlo = 0
		ylo = 0
		zlo = 0

		xlo_bound = xlo + min(0.0,xy,xz,xy+xz)
		xhi_bound = lx + xlo + max(0.0,xy,xz,xy+xz)
		ylo_bound = ylo + min(0.0,yz)
		yhi_bound = ly + ylo + max(0.0,yz)
		zlo_bound = zlo
		zhi_bound = lz + zlo

		natoms = len(images[k])
		x = pos[:,0]
		y = pos[:,1]
		z = pos[:,2]
		symbols = images[k].get_chemical_symbols()
		typ = np.array([E[sym] for sym in symbols])
		ids = np.array([d+1 for d in range(len(symbols))])
		
		dfile.write("ITEM: TIMESTEP\n")
		dfile.write("%d\n" % k)
		dfile.write("ITEM: NUMBER OF ATOMS\n")
		dfile.write("%d\n" % natoms)
		dfile.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
		dfile.write("%f %f %f\n" % (xlo_bound, xhi_bound, xy) )
		dfile.write("%f %f %f\n" % (ylo_bound, yhi_bound, xz) )
		dfile.write("%f %f %f\n" % (zlo_bound, zhi_bound, yz) )
		dfile.write("ITEM: ATOMS id type x y z\n")
		for at in range(natoms):
			dfile.write("%d %d %f %f %f\n" % (ids[at],typ[at],x[at],y[at],z[at]))
		
		xmin = min(x)
		xmax = max(x)
		xrng = xmax-xmin
		ymin = min(y)
		ymax = max(y)
		yrng = ymax-ymin
		zmin = min(z)
		zmax = max(z)
		zrng = zmax-zmin

		bw=1.85*fudge_fact+0.1 #Use smallest possible bin size
		nbinats=15
		ax=bw #in A
		ay=bw #in A
		az=bw #in A
		nx=int((xmax - xmin)/ax)# + 1
		ny=int((ymax - ymin)/ay)# + 1
		nz=int((zmax - zmin)/az)# + 1
		totats=zeros([nx,ny,nz], int)
		atinfo=zeros([nx,ny,nz,nbinats,5], float)

		for i in range(natoms):
			b1=int((x[i]-xmin)/ax)
			b2=int((y[i]-ymin)/ay)
			b3=int((z[i]-zmin)/az)
			if(b1>=nx):b1=nx-1
			if(b2>=ny):b2=ny-1
			if(b3>=nz):b3=nz-1
			atinfo[b1][b2][b3][totats[b1][b2][b3]][0]=ids[i]
			atinfo[b1][b2][b3][totats[b1][b2][b3]][1]=typ[i]
			atinfo[b1][b2][b3][totats[b1][b2][b3]][2]=x[i]
			atinfo[b1][b2][b3][totats[b1][b2][b3]][3]=y[i]
			atinfo[b1][b2][b3][totats[b1][b2][b3]][4]=z[i]
			totats[b1][b2][b3]+=1

		nbin=nx*ny*nz+1             #nominal value to initialise arrays

		ofile.write("# Timestep %s \n" %(k))
		ofile.write("# \n")
		ofile.write("# Number of particles %s \n" %(natoms))
		ofile.write("# \n")
		ofile.write("# Max number of bonds per atom 4 with coarse bond order cutoff 0.300 \n")
		ofile.write("# Particle connection table and bond orders \n")
		ofile.write("# id type nb id_1...id_nb \n")
		for b1 in range(nx):
			#print(b1," of ", nx, '\n')
			for b2 in range(ny):
				for b3 in range(nz):
					for l in range(totats[b1][b2][b3]):
						alist=[]
						alist.append(int(atinfo[b1][b2][b3][l][0]))#Append id and type of atom in question to the list
						alist.append(int(atinfo[b1][b2][b3][l][1]))
						for xi in range(-1,2):
							for yi in range(-1,2):
								for zi in range(-1,2):
									B1=b1+xi
									B2=b2+yi
									B3=b3+zi
									if(B1>=nx):B1=B1%nx
									if(B2>=ny):B2=B2%ny
									if(B3>=nz):B3=B3%nz
									for k in range(totats[B1][B2][B3]):
										dist=dists(xrng,yrng,zrng,atinfo[B1][B2][B3][k][2],atinfo[b1][b2][b3][l][2],atinfo[B1][B2][B3][k][3],atinfo[b1][b2][b3][l][3],atinfo[B1][B2][B3][k][4],atinfo[b1][b2][b3][l][4])
										lookup=str(int(atinfo[b1][b2][b3][l][1]))+str(int(atinfo[B1][B2][B3][k][1]))
										if(dist<=float(rdict[lookup])*fudge_fact and int(atinfo[b1][b2][b3][l][0])!=int(atinfo[B1][B2][B3][k][0])): alist.append(int(atinfo[B1][B2][B3][k][0]))
						  
						alist.insert(2, len(alist)-2) #Insert number of bonded atoms in col 3
						outstr=' '.join(map(str, alist))+"\n" #Write list of connected atoms
						ofile.write(outstr)
		ofile.write("# \n")
	ofile.write("# ")
	ofile.close()

	return elements
	

def reaxFF_botable(trajectory, working_dir, interval, ffield, ffoption, lmp_path, atomsk_path):

	os.chdir(working_dir)
	images = read(trajectory,index=':')

	bo_string  = ""
	dump_string = ""
	s1 = []
	for i in range(0,len(images),interval):
		atoms = images[i]
		chem = atoms.get_chemical_symbols()
		elements = sorted(list(set(chem)), key=lambda x: atomic_numbers[x])
		atom_types = {}
		for el, j in zip(elements, range(len(elements))):
			atom_types[el] = j+1

		chargefile('charge.txt', elements)
		input_file2('base.in',elements,ffield,ffoption)
		basename = str(i+1)
		xyz_filename = basename + '.reax'
		write_lammps_data(filename=xyz_filename, atoms=atoms, atom_types=atom_types, units='real')
		if i == 0:
			write_lammps_data(filename='data.in', atoms=atoms, atom_types=atom_types, units='real')
		atomsk(basename, atomsk_path, xyz_filename)
		lammps(basename, lmp_path, 1)

		#read bo file of each image
		bo_out = 'bonds.reaxc_%d' % (i+1)
		dump_out = 'min.dump_%d' % (i+1)
		md_out = 'lmp.out'

		r_bo = open(bo_out, 'r').readlines()
		t1 = r_bo[0].split()
		t1[2] = str(i+1) + "\n"
		r_bo[0] = " ".join(t1)
		r_dump = open(dump_out).readlines()
		r_dump[1] = str(i+1) + "\n"

		bo_string += "".join(r_bo)
		dump_string += " ".join(r_dump)

	with open('bonds.reaxc','w') as f1:
		f1.write(bo_string)
	with open('dump_reax.dump','w') as f2:
		f2.write(dump_string)

	subprocess.call('rm *.lmp *.reax force.dump lmp.out min.dump_* bonds.reaxc_*', shell=True)
	return elements

"""  
rdict = {'11':'1.30',
		 '22':'1.55', 
		 '33':'1.60', 
		 '44':'1.60',
		 '12':'1.20', 
		 '21':'1.20',
		 '23':'1.65',
		 '32':'1.65',
		 '24':'1.65', 
		 '42':'1.65',
		 '13':'1.30',
		 '31':'1.30', 
		 '14':'1.30',
		 '41':'1.30',
		 '34':'1.60',
		 '43':'1.60'}
"""

def Dynamics_species(trajectory, working_dir, interval, ffield_reax, ffoption, exclude, lmp_path, atomsk_path, rdict, ReaxFFBO=True):
	E = {}
	if ReaxFFBO:
		elements = reaxFF_botable(trajectory, working_dir, interval, ffield_reax, ffoption, lmp_path, atomsk_path)
		for i, e in enumerate(elements):
			E[str(i+1)] = e
	else:
		elements = Rcutoff_botable(trajectory, working_dir, interval, rdict)
		for i, e in enumerate(elements):
			E[str(i+1)] = e
	output = Species_Rec('bonds.reaxc', 'dump_reax.dump',E,'data.in', exclude)

	return output


def Large_to_Small(trajectory, working_dir, interval, ffield_reax, ffoption, lmp_path, atomsk_path, debug=False):
	#Read trajectory and find cell information
	os.chdir(working_dir)
	images = read(trajectory,index=':')
	#################
	bo_string  = ""
	dump_string = ""
	s1 = []
	for i in range(0,len(images),interval):
		atoms = images[i]
		chem = atoms.get_chemical_symbols()
		elements = sorted(list(set(chem)), key=lambda x: atomic_numbers[x])
		atom_types = {}
		for el, j in zip(elements, range(len(elements))):
			atom_types[el] = j+1

		chargefile('charge.txt', elements)
		input_file2('base.in',elements,ffield_reax,ffoption)
		basename = str(i+1)
		xyz_filename = basename + '.reax'
		write_lammps_data(filename=xyz_filename, atoms=atoms, atom_types=atom_types, units='real')
		if i == 0:
			write_lammps_data(filename='data.in', atoms=atoms, atom_types=atom_types, units='real')

		datafile = 'data.in'
		atomsk(basename, atomsk_path, xyz_filename)
		lammps(basename, lmp_path, 1)

		#read bo file of each image
		bo_out = 'bonds.reaxc_%d' % (i+1)
		dump_out = 'min.dump_%d' % (i+1)
		md_out = 'lmp.out'

		r_bo = open(bo_out, 'r').readlines()
		t1 = r_bo[0].split()
		t1[2] = str(i+1) + "\n"
		r_bo[0] = " ".join(t1)
		r_dump = open(dump_out).readlines()
		r_dump[1] = str(i+1) + "\n"

		bo_string += "".join(r_bo)
		dump_string += " ".join(r_dump)

	with open('bonds.reaxc','w') as f1:
		f1.write(bo_string)
	with open('dump_reax.dump','w') as f2:
		f2.write(dump_string)
	subprocess.call('rm *.lmp *.reax force.dump lmp.out min.dump_* bonds.reaxc_*', shell=True)

	E = {}
	for i, e in enumerate(elements):
		E[str(i+1)] = e
	#################
	DFfile = 'dump_reax.dump'
	BOfile = 'bonds.reaxc'
	exclude = []

	atom_types = extract_atom_types(datafile)
	t1_list = read_BOTable(BOfile)
	t_list, t_info = t1_list
	cluster_str = cluster_list_to_molecule_str(atom_types, t_list)
	print(cluster_str)
	filelist = extract_coordinate(DFfile, BOfile, cluster_str, types=E, t_info=t_info, exclude=exclude)

	# Split and collect species for smaller cell
	original = copy.deepcopy(images)

	for i in range(len(filelist)):
		single_time = read(filelist[i],index=":")
		cell = single_time[0].get_cell()[:]
		xn = math.ceil(cell[0][0] / 15)
		yn = math.ceil(cell[1][1] / 15)
		zn = math.ceil(cell[2][2] / 15)

		#print(xn,yn,zn)
		xl = cell[0]/xn
		yl = cell[1]/yn
		zl = cell[2]/zn

		atoms0_collect = []
		atoms1_collect = []
		for x in range(xn+1):
			for y in range(yn+1):
				for z in range(zn+1):
					d = [ [x*xl[0], (x+1)*xl[0]], [y*yl[1], (y+1)*yl[1]], [z*zl[2], (z+1)*zl[2]]]
					cell2 = copy.deepcopy(cell)
					cell2[0][0] = d[0][1] - d[0][0] #+ xl[0]*0.5*x
					cell2[1][1] = d[1][1] - d[1][0] #+ yl[1]*0.5*y
					cell2[2][2] = d[2][1] - d[2][0] #+ zl[2]*0.5*z
					shift =  np.array([xl[0]*0.5*x, yl[1]*0.5*y, zl[2]*0.5*z])
					images = []
					for m in range(len(single_time)):
						molecule = single_time[m]
						pos = molecule.get_positions() - shift
						molecule2 = copy.deepcopy(molecule)
						if debug != True:
							molecule2.set_positions(pos)
							molecule2.set_cell(cell2)
						testp1= (pos[:,0] <= cell2[0][0]).all()
						testp2= (pos[:,1] <= cell2[1][1]).all()
						testp3= (pos[:,2] <= cell2[2][2]).all()
						test2 = (pos >= -0.1).all()
						if test2 and testp1 and testp2 and testp3:
							images.append(molecule2)
					#print(len(images))
					atoms0 = images[0]
					for im in range(1,len(images)):
						atoms0 += images[im]
					#view(atoms0)
					atoms0_collect.append(atoms0)

					
					atoms_test = []
					for a in range(len(original[0])):

						pos2 = original[0][a].position - shift
						otestp1= (pos2[0] <= cell2[0][0]-1).all()
						otestp2= (pos2[1] <= cell2[1][1]-1).all()
						otestp3= (pos2[2] <= cell2[2][2]-1).all()
						test3 = (pos2 >= 0).all()
						atom = copy.deepcopy(original[0][a])
						atom.position = pos2
						if test3 and otestp1 and otestp2 and otestp3:
							if debug == True:
								atoms_test.append(original[0][a])
							else:
								atoms_test.append(atom)

					symbol = ''
					positions = []
					for a in range(len(atoms_test)):
						symbol += atoms_test[a].symbol
						positions.append(atoms_test[a].position)

					if debug == True:
						atoms_section = Atoms(symbol,positions=positions,cell=original[0].get_cell())
					else:
						atoms_section = Atoms(symbol,positions=positions,cell=cell2)
					atoms1_collect.append(atoms_section)
					
			write('atoms0_%d.xyz' % i,atoms0_collect, format='xyz')
			write('atoms1_%d.xyz' % i,atoms1_collect, format='xyz')





	return

################################################################################
def Species_Rec(BOfile, DFfile, types, datafile='',exclude=[]):

    start = time.time()

    atom_types = extract_atom_types(datafile)
    t1_list = read_BOTable(BOfile)
    #examine_element_neighbor(BOfile, types, 'H', 2)
    t_list, t_info = t1_list
    cluster_str = cluster_list_to_molecule_str(atom_types, t_list)
    filelist = extract_coordinate(DFfile,BOfile, cluster_str, types=types, t_info=t_info, exclude=exclude)
    #np.savetxt('species_ids.dat', cluster_str[0], fmt='%s')
    #CHNO atom type
    #times,ids,unsort_ids = species_timeframes(cluster_str,'C5H8N4O12')
    #np.savetxt('times_bonded.dat', unsort_ids, fmt='%s')
    #appearance_timeframes = print_lines(times,ids)
    #np.savetxt('PETN_times_id.dat', appearance_timeframes, fmt='%s')

    end = time.time()
    #print("Total time: {}".format(end-start))

    return filelist

################################################################################

def species_dft_dynamics(base_name, 
						 base_dir, 
						 dft_input_dir, 
						 lmp_job_done, 
						 ncores, queue, walltime, 
						 scheduler, 
						 exclude, 
						 interval, 
						 ffield_reax, 
						 ffoption, 
						 lmp_path, 
						 atomsk_path, 
						 modules,
						 rdict, 
						 ReaxFFBO,
						 python_path='python'):
	os.chdir(base_dir)
	ffield_reax = ffield_reax
	dft_species = base_dir +'/5_DFT_species/species_dft_dyn_'+str(base_name)
	qnums = {}
	subdir = {}
	try:
		os.stat(dft_species)
	except:
		os.makedirs(dft_species)

	total_xyz_files = []
	total_bo_files = []
	species_xyz = []
	jobname = 'DFT_species_dyn'
	for qnum in lmp_job_done.keys():
		path = lmp_job_done[qnum][0]
		Ndump = lmp_job_done[qnum][1]
		cal_name = path.split('/')[-1]
		species_subdir = dft_species + '/' + cal_name
		try:
			os.stat(species_subdir)
		except:
			os.mkdir(species_subdir)
			subprocess.call('cp '+path+'/dynamics_%d.xyz' % Ndump+' '+species_subdir,shell=True)
			os.chdir(species_subdir)
			Dynamics_species(trajectory='dynamics_%d.xyz' % Ndump,
							working_dir=species_subdir,ffield_reax=ffield_reax,
							ffoption=ffoption,
							interval=interval,
							exclude=exclude, lmp_path=lmp_path, atomsk_path=atomsk_path,
							rdict=rdict, ReaxFFBO=ReaxFFBO)
		total_xyz_files.append(species_subdir+'/total.xyz')
		total_bo_files.append(species_subdir+'/total_bo.reaxc')
	species_dir = dft_species+'/species'

	if os.path.exists(species_dir):
		jobs = []
		subdirs = []
		if os.path.exists(species_dir+'/submit'):
			with open(species_dir+'/submit','r') as submit:
				contents = submit.readlines()
			for line in contents:
				line_ = line.split()
				print(line_)
				job = line_[0]
				path0 = line_[1]
				qnums[job] = path0
				jobs.append(job)
			subdir[species_dir] = jobs
		return qnums, subdir, lmp_job_done

	else:
		os.mkdir(species_dir)
		os.chdir(species_dir)
		subprocess.call('cp '+dft_input_dir+'/species_dynamics.py'+' '+species_dir,shell=True)

		com = 'cat '
		com2 = 'cat '
		for i in range(len(total_xyz_files)):
			com += total_xyz_files[i] + ' '
			com2 += total_bo_files[i] + ' '
		com += '> '+species_dir+'/species.xyz'
		com2 += '> '+species_dir+'/species_bo.reaxc'
		subprocess.call(com,shell=True)
		subprocess.call(com2,shell=True)


		if ReaxFFBO:
			images, botables = sort_species(xyzfile=species_dir+'/species.xyz', bofile=species_dir+'/species_bo.reaxc')
			write(species_dir+'/species_sorted.xyz',images)
			with open(species_dir+'/species_sorted_bo.reaxc','w') as sbo:
				bo_str = "".join(botables)
				sbo.write(bo_str)
			bond_infos = bond_analysis(images, botables)
			write(species_dir+'/species_reduced.xyz',bond_infos[:,0])

		else:
			images = read('species.xyz',index=':')
			images_sorted = sorted(images, key=lambda x:len(x))
			images_reduced = [atoms for atoms in images_sorted if len(atoms) > 1 and len(atoms) < 100]
			write(species_dir+'/species_reduced.xyz',images_reduced)

		job_submit_file(path=species_dir,
						ncores=ncores,
						walltime=walltime,
						queue=queue,
						commands=[python_path+' species_dynamics.py'],
						jobname='species_md',
						modules=modules,
						scheduler='slurm')

		jobs = []
		subdirs = []
		if not os.path.exists(species_dir+'/submit'):
			trajectory = read('species_reduced.xyz', index=':')
			trajectory = list(filter(lambda x: len(x) <= 100, trajectory))
			for i in range(0,len(trajectory)):
				os.chdir(species_dir)
				os.mkdir(str(i+1))
				if scheduler == 'slurm':
					subprocess.call('cp species_dynamics.py job.slurm '+str(i+1), shell=True)
				elif scheduler == 'pbs':
					subprocess.call('cp species_dynamics.py job.pbs '+str(i+1), shell=True)
				os.chdir(species_dir+'/'+str(i+1))
				subdirs.append(species_dir+'/'+str(i+1))
				trajectory[i].write('POSCAR_1')
				path0 = os.getcwd()
				if scheduler == 'slurm':
					len_qnum0 = 0
					while len_qnum0 == 0:
						p = subprocess.Popen(['sbatch','job.slurm'],stdout=subprocess.PIPE)
						qnum0 = p.communicate()[0].decode("utf-8").split('.')[0]
						len_qnum0 = len(qnum0)
					if len_qnum0 != 0:
						qnums[qnum0.split()[-1]] = path0
						jobs.append(qnum0.split()[-1])

				elif scheculer == 'pbs':
					p = subprocess.Popen(['qsub','job.pbs'],stdout=subprocess.PIPE)
					qnum0 = p.communicate()[0].decode("utf-8")
					qjob = qnum0.split('.')[0]

					if len(qnum0) != 0:
						qnums[qjob] = path0
						jobs.append(qjob)
			subdir[species_dir] = jobs
			with open(species_dir+'/submit','w') as submit:
				for job, path in zip(jobs, subdirs):
					submit.write("%s %s\n" % (job, path))
		elif os.path.exists(species_dir+'/submit'):
			with open(dft_sp_subdir+'/submit','r') as submit:
				contents = submit.readlines()
				for line in contents:
					line_ = line.split()
					job = line_[0]
					path0 = line_[1]
					qnums[job] = path0
					jobs.append(job)
				subdir[species_dir] = jobs
		os.chdir(species_dir)
		#del lmp_job_done[qnum]

		return qnums, subdir, lmp_job_done
			

def Search_Bonding_species(trajectory, center_symbol, neighborcutoff):

	from ase.visualize import view
	atoms = read(trajectory, index=':')
	for m in range(len(atoms)):
		nl = NeighborList([neighborcutoff]*len(atoms[m]))
		nl.update(atoms[m])
		for j in range(len(atoms[m])):
			if atoms[m][j].symbol == center_symbol:
				indices, offsets = nl.get_neighbors(m)
				if len(indices) == 2:
					print(atoms[m])

	return 


def Analyze_database(database):

	Nimages = 0
	images = []
	chems = []
	with open(database,'r') as input_data:
		contents = input_data.readlines()
	for i in range(len(contents)):
		line = contents[i]
		if line.split()[0] == 'begin':
			Sindex = i
		elif line.split()[0] == 'end':
			Nimages += 1
			Eindex = i
			eachimage = contents[Sindex:Eindex]
			lattice = np.array([l.split() for l in eachimage[2:5]])
			atoms = np.array([a.split() for a in eachimage[5:-2]])
			energy = float(eachimage[-2].split()[1])
			cell = lattice[:,1:]
			cell = cell.astype(np.float)
			positions = atoms[:,1:4]
			positions = positions.astype(np.float)
			forces = atoms[:,-3:]
			forces = forces.astype(np.float)
			symbols =[c[0] for c in atoms[:,4:5]]
			formula = "".join(symbols)
			atoms_object = Atoms(formula, cell=cell,
					positions=positions, pbc=[1,1,1])
			atoms_object.set_calculator(SPC(atoms_object, energy = energy,
						                    forces=forces))
			images.append(atoms_object)
			chems.append(atoms_object.get_chemical_formula(mode='hill'))

	exclude = list(set(chems))

	return exclude


def xyz_to_smiles(fname):
    mol = next(pybel.readfile("xyz", fname))
    smi = mol.write(format="smi")
    return smi.split()[0].strip()


def sort_species(xyzfile,bofile):
	images = read(xyzfile,index=':')
	botables = []
	count = 0
	with open(bofile,'r') as bo:
		lines = bo.readlines()
	for i in range(len(lines)):
		line0 = lines[i].split()
		if len(line0) > 2 and line0[1] == 'Timestep':
			Natoms = int(lines[i+2].split()[-1])
			bo_block = lines[i:i+6+Natoms+2]
			botables.append(bo_block)

	conjugate = []
	for i in range(len(images)):
		atoms = images[i]
		botable = botables[i]
		conjugate.append([atoms,botable])

	conjugate_sorted = sorted(conjugate, key=lambda x:sum(x[0].get_masses()))
	for i in range(len(conjugate_sorted)):
		lines = conjugate_sorted[i][1]
		Original_index = []
		Natoms = int(lines[2].split()[-1])
		start = 7
		header = lines[0:7]
		block = lines[start:start+Natoms]
		for atom in block:
			a = re.split(r'(\s+)', atom)
			Original_index.append(a[2])

		D = {}
		for index in range(len(Original_index)):
			D[Original_index[index]] = str(index+1)

		new_lines = []
		for atom in block:
			a = re.split(r'(\s+)', atom)
			new_ind = D[a[2]]
			a[2] = new_ind
			Nneigh = int(a[6])
			for n in range(Nneigh):
				new_nn_ind = D[a[8+n*2]]
				a[8+n*2] = new_nn_ind

			new_lines.append("".join(a))

		new_header = "".join(header)
		new_block = "".join(new_lines) +"#"
		string = new_header+new_block
		conjugate_sorted[i][1] = string

	output = np.array(conjugate_sorted)
	images = output[:,0]
	botables = output[:,1]
	return images, botables

def bond_analysis(images, botables):
	bond_infos = []
	with open('bond_analysis_out.txt','w') as f:
		f.write("Natoms, Molecular Mass, Bond analysis\n")
		for i in range(len(botables)):
			string = ""
			atoms = images[i]
			botable = botables[i]
			mass = sum(atoms.get_masses())
			bond_info = bond_analysis_single(botable)
			f.write("%d\t%6.4f " % (len(atoms),mass))
			string += "%d\t%6.4f " % (len(atoms),mass)
			for key in bond_info.keys():
				f.write("\t%s:%d, " % (key, bond_info[key]))
				string += "\t%s:%d, " % (key, bond_info[key])
			f.write("\n")

			bond_infos.append([atoms,string])

	new_bond_infos = np.array([bond_infos[0]])
	for i in range(1,len(bond_infos)):
		if bond_infos[i][1] not in list(new_bond_infos[:,1]):
			new_bond_infos = list(new_bond_infos)
			new_bond_infos.append(bond_infos[i])
		new_bond_infos = np.array(new_bond_infos)

	return new_bond_infos

def bond_analysis_single(botable):
	geom = 280
	nneigh=10
	natt=4 #number of atom types
	bc=zeros((natt,natt),float)
	t_start=0
	nats=0
	ifile = botable.splitlines(True)
	for line in ifile:
		items = line.split(" ")
		if(items[1] == "Timestep"):
			t_start=int(items[2])
		if(items[1] == "Number"):
			nats=int(items[4])
			break
	#num_lines = sum(1 for line in open(input_file))
	num_lines = len(ifile)
	nts = int(num_lines/(nats+8))
	#print("Number of time frames = "+str(nts))

	in_prm=np.zeros((2,),dtype=float) #starting times x timesteps
	in_prm[0]=t_start
	in_prm[1]=0.1 #TIMESTEP

	dump_file='coordsum.txt'
	output_file='coordmat.txt'
	#print("Using simple 0.5 bond cutoff")
	bc[0,0]=0.50 # H-H
	bc[0,1]=0.50 # C-H
	bc[1,0]=0.50 # C-H
	bc[0,2]=0.50 # N-H
	bc[2,0]=0.50 # N-H
	bc[0,3]=0.50 # O-H
	bc[3,0]=0.50 # O-H
	bc[1,1]=0.50 # C-C
	bc[1,2]=0.50 # C-N
	bc[2,1]=0.50 # C-N
	bc[1,3]=0.50 # C-O
	bc[3,1]=0.50 # C-O
	bc[2,2]=0.50 # N-N
	bc[2,3]=0.50 # N-O
	bc[3,2]=0.50 # N-O
	bc[3,3]=0.50 # O-O

	atomid=zeros((nts,nats,2),int)
	bondn=zeros((nts,nats,nneigh),int)
	bonds=zeros((nts,nats,nneigh),float)
	geoms=zeros((nts,natt,5,5,5,5,nneigh),int)
	stuff=zeros((nts,nats,3),float)
	tstep=zeros((nts),float)
	t_a=zeros((nts),float)
	bndn=zeros(nneigh,int)
	bnds=zeros(nneigh,float)
	j=0
	t=-1
	natoms=0
	nbonds=0
	nlines=nats+8
	tbonds=0

	#Read data file
	#ifile=open(input_file, 'r')
	ifile = botable.splitlines(True)
	for line in ifile:
		tmp=re.split(r'[\s]\s*', line)
		if((j%nlines)==0):
			t+=1
			if(t>=nts):
				t-=1
				break
			initial_t=eval(tmp[2])
			if(t==0):
				t_a[t]=0
			else:
				t_a[t]=(initial_t-t_start)
				tstep[t]=(t_a[t]-t_a[t-1])*t*in_prm[1]/1000.
		if((j%nlines)==2):
			natoms=eval(tmp[4])
		if((j%nlines)==4):
			nbonds=eval(tmp[7])
			if (nbonds>nneigh):
				print ("oops.  exceeded number of neighbors")
		if(((j%nlines)>6) and ((j%nlines)<nats+7)):
			i=eval(tmp[1])-1 # atom number offset to 0:nats-1
			n=eval(tmp[2])
			k=eval(tmp[3])
			atomid[t,i,0]=n # atom type 1-4
			atomid[t,i,1]=k # number of bonds to that atom
			for kk in range(nneigh): # need to zero these out for sort operation
				bndn[kk]=0
				bnds[kk]=0.
			for kk in range(k):
				bndn[kk]=eval(tmp[kk+4])-1 # atom numbers of bonded atoms offset to 0:nats-1
				bnds[kk]=eval(tmp[kk+5+k]) # bond orders for bonded atoms
				tbonds+=1
			for kk in range(3):
				stuff[t,i,kk]=eval(tmp[k+k+5+kk])
			ids=argsort(bnds)
			for kk in range(k):
				bondn[t,i,kk]=bndn[ids[nneigh-1-kk]]
				bonds[t,i,kk]=bnds[ids[nneigh-1-kk]]
		j+=1
	#ifile.close()
	#print ("lines read in = %d" %(j-8))
	#print ("timesteps read in = %d" %(t+1))
	#print ("total bonds read in = %d" %(tbonds))

	ts=t+1
	ofile=open(output_file, 'w')
	bondt=zeros((nts,natt,natt),int)
	for t in range(ts):
		kkmax=0
		tbonds=0
		for i in range(nats):
			ca=atomid[t,i,0]-1  # atom type 0-3
			k=atomid[t,i,1] # number of bonds to that atom
			cn=zeros(5,int) # numbers of each atom type or nothing bound to atom
			kkn=0
			kk=k
			k=min(k,4)
			for kk in range(k):
				nn=atomid[t,bondn[t,i,kk],0]-1
				if (bonds[t,i,kk]>bc[ca,nn]):
					cn[nn]+=1
					bondt[t,ca,nn]+=1
					kkn+=1
					cnnn=cn[nn]
					tbonds+=1
			kkmax=max(kkmax,kkn)
			cn[4]=nneigh-1-cn[0]-cn[1]-cn[2]-cn[3]
			geoms[t,ca,cn[0],cn[1],cn[2],cn[3],cn[4]]+=1
		tbonds/=2 # correct for double counting
		bondt[t,0,0]/=2 # correct for double counting
		bondt[t,1,1]/=2 # correct for double counting
		bondt[t,2,2]/=2 # correct for double counting
		bondt[t,3,3]/=2 # correct for double counting
		ofile.write('%d %d %d %d %d %d %d %d %d %d %d %d\n' %(tstep[t],tbonds,bondt[t,0,0],bondt[t,0,1],bondt[t,0,2],bondt[t,0,3],bondt[t,1,1],bondt[t,1,2],bondt[t,1,3],bondt[t,2,2],bondt[t,2,3],bondt[t,3,3]))
	ofile.close

	hdr_for='type'
	for t in range(ts):
	   hdr_for+=' %0.3f'
	hdr_for+=' \n'
	hdr_str='tstep[0]'
	for t in range(ts-1):
	   hdr_str+=', tstep[%d]' %(t+1)

	A_str=("H","C","N","O")
	vec_for='%s'
	for t in range(ts):
	   vec_for+=' %d'
	vec_for+=' \n'
	vec_str='label'
	for t in range(ts):
		vec_str+=', dum[%d]' %(t)

	dum=zeros(nts,float)
	Collect = {}
	ofile=open('coord.txt', 'w')
	ofile.write(hdr_for %(eval(hdr_str)))
	for i in range(4):
		for cn in range(5):
			for c1 in range(cn+1):
				for c2 in range(cn+1-c1):
					for c3 in range(cn+1-c1-c2):
						c4=cn-c1-c2-c3
						c5=nneigh-1-cn
						dum[:]=geoms[:,i,c1,c2,c3,c4,c5]
						label='%s[' %(A_str[i])
						for cc in range(c1+1):
							if (cc>0):
								label+='H'
						for cc in range(c2+1):
							if (cc>0):
								label+='C'
						for cc in range(c3+1):
							if (cc>0):
								label+='N'
						for cc in range(c4+1):
							if (cc>0):
								label+='O'
						label+=']'
						ofile.write(vec_for %(eval(vec_str)))
						#Collect[label] = int(dum[0])
						if int(dum[0]) != 0:
							Collect[label] = int(dum[0])

	ofile.close

	return Collect


