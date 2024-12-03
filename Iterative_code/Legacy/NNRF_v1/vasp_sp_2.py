#!/bin/env python
import subprocess, copy
import numpy as np
from .lammpslib import write_lammps_data
from .tracking import job_submit_file
from ase.io import read, write
from ase.data import atomic_numbers, atomic_masses
import sys, os, glob, re, json
from ase import Atoms, Atom
from ase.db import connect
from ase.visualize import view
import collections
import pandas as pd
from ase.units import eV, kcal, mol
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from shutil import copyfile
from ase.io import read, write
import time
from itertools import combinations


def atomsk(basename, atomsk_path, xyz_filename):
	atomsk_command = atomsk_path +  " " + xyz_filename + " lmp -properties charge.txt"
	subprocess.call(atomsk_command, shell=True)
	return

def lammps(lmp_filename,lmp_path, ncores):
	lammps_command ="mpiexec -np "+str(ncores)+" " + lmp_path + " -var filename " + lmp_filename + " -in base.in > lmp.out"
	subprocess.call(lammps_command, shell=True)
	return

def chargefile(filename, elements):
	with open(filename, 'w') as c1:
		c1.write('charge\n')
		elements = list(set(elements))
		for element in elements:
			c1.write(element + ' 0.0\n')

def zbl_input_file(filename, elements):
	#e_string = " ".join(elements)
	#print(e_string)
	type_vs_AN = {}
	for i, element in enumerate(elements):
		type_vs_AN[i+1] = atomic_numbers[element]

	l = [j for j in list(type_vs_AN.keys())]
	comb = list(combinations(l,2))
	for i in range(len(elements)):
		comb.append( (i+1,i+1) )

	pair_coeffs = []
	for pair in comb:
		string0 = 'pair_coeff %d %d %d %d\n' % (pair[0], pair[1], type_vs_AN[pair[0]], type_vs_AN[pair[1]])
		pair_coeffs.append(string0)
	

	with open(filename,'w') as ff:
		string = "units           real\n"
		string += "boundary        p p p\n"
		string += "atom_style      charge\n"
		string += "neighbor        2.0 nsq\n"
		string += "neigh_modify    delay 2\n"
		string += "box tilt large\n"
		string += "read_data       ${filename}.data\n\n"

		string += "#    Potential Parameters\n"
		string += "pair_style      zbl 3.5 4.5\n"
		for pair_coeff in pair_coeffs:
			string += pair_coeff
		string += "thermo_style    custom step pe pxx pyy pzz pxy pxz pyz\n"
		string += "thermo   1\n"
		string += "dump            d1 all custom 1 force.dump fx fy fz\n"
		string += "dump_modify     d1 sort id\n"
		string += "run           0\n"
		ff.write(string)

	return

def input_file(filename, elements, ffield):
	e_string = " ".join(elements)
	print(e_string)

	with open(filename,'w') as ff:
		string = "units           real\n"
		string += "boundary        p p p\n"
		string += "atom_style      charge\n"
		string += "neighbor        2.0 nsq\n"
		string += "neigh_modify    delay 2\n"
		string += "box tilt large\n"
		string += "read_data       ${filename}.reax\n\n"

		string += "#    Potential Parameters\n"
		string += "pair_style      reax/c NULL  safezone 4.0 mincap 400\n"
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
		string += "dump            d1 all custom 1 force.dump fx fy fz\n"
		string += "dump_modify     d1 sort id\n"
		string += "run           0\n"
		ff.write(string)

	return

def dft_sp_nitramine(base_name, base_dir, job_done, interval, dft_input_dir, ncores, queue, walltime, scheduler, modules, python_path='python'):
	os.chdir(base_dir)
	##dft_sp_dir = base_dir+'/4_DFT_sp/DFT_sp_'+str(base_name)
	dft_sp_dir = base_dir+'/DFT_sp_'+str(base_name)
	try:
		os.stat(dft_sp_dir)
	except:
		os.makedirs(dft_sp_dir)

	tmp_num = 0
	qnums = {}
	dft_subdir = {}
	jobname = 'DFT_sp'
	job_done2 = copy.deepcopy(job_done)
	for qnum in job_done.keys():
		path = job_done[qnum][0]
		Ndump = job_done[qnum][1]
		cal_name = path.split('/')[-1]
		dft_sp_subdir = dft_sp_dir + '/' + cal_name
		try:
	 		os.stat(dft_sp_subdir)
		except:
			os.mkdir(dft_sp_subdir)
			subprocess.call('cp '+path+'/dynamics_%d.xyz' % Ndump+' '+dft_sp_subdir,shell=True)
			subprocess.call('cp '+dft_input_dir+'/dft_sp.py'+' '+dft_sp_subdir,shell=True)

			job_submit_file(path=dft_sp_subdir,
						    ncores=ncores,
						    walltime=walltime,
						    queue=queue,
						    commands=[python_path+' dft_sp.py'],
						    jobname='DFT_sp',
						    modules=modules,
						    scheduler='slurm')

		os.chdir(dft_sp_subdir)
		jobs = []
		subdirs = []
		if not os.path.exists(dft_sp_subdir+'/submit'):
			xyzf = glob.glob('dynamics_%d.xyz' % Ndump)
			trajectory = read('dynamics_%d.xyz' % Ndump, index=':')
			for i in range(0,Ndump,interval):
				os.chdir(dft_sp_subdir)
				os.mkdir(str(i+1))
				if scheduler == 'slurm':
					subprocess.call('cp dft_sp.py job.slurm '+str(i+1), shell=True)
				elif scheduler == 'pbs':
					subprocess.call('cp dft_sp.py job.pbs '+str(i+1), shell=True)
				os.chdir(dft_sp_subdir+'/'+str(i+1))
				subdirs.append(dft_sp_subdir+'/'+str(i+1))
				trajectory[i].write('POSCAR_1')
				path0 = os.getcwd()
				if scheduler == 'slurm':
					len_qnum0 = 0
					while len_qnum0 == 0:
						p = subprocess.Popen(['sbatch','job.slurm'],stdout=subprocess.PIPE)
						qnum0 = p.communicate()[0].decode("utf-8").split('.')[0]
						len_qnum0 = len(qnum0)
						print(len_qnum0)

					if len_qnum0 != 0:
						qnums[qnum0.split()[-1]] = path0
						jobs.append(qnum0.split()[-1])

				elif scheduler == 'pbs':
					p = subprocess.Popen(['qsub','job.pbs'],stdout=subprocess.PIPE)
					qnum0 = p.communicate()[0].decode("utf-8") # qsub: submit error (Maximum number of jobs already in queue for user MSG=total number of current user's jobs exceeds the queue limit: user yoo    p@halstead-fe00.rcac.purdue.edu, queue standby)
				#deal with submission error
					qjob = qnum0.split('.')[0]
					job_flag = qjob.isdigit()
					while not job_flag:
						p = subprocess.Popen(['qsub','job.pbs'],stdout=subprocess.PIPE)
						qnum0 = p.communicate()[0].decode("utf-8")
						qjob = qnum0.split('.')[0]
						job_flag = qjob.isdigit()

					if len(qnum0) != 0:
						qnums[qnum0.split('.')[0]] = path0
						jobs.append(qnum0.split('.')[0])

			dft_subdir[dft_sp_subdir] = jobs
			with open(dft_sp_subdir+'/submit','w') as submit:
				for job, path in zip(jobs, subdirs):
					submit.write("%s %s\n" % (job, path))
		elif os.path.exists(dft_sp_subdir+'/submit'):

			#with open(dft_sp_subdir+'/submit','r') as submit:
			#	contents = submit.readlines()
			#	for line in contents:
			#		line_ = line.split()
			#		job = line_[0]
			#		path0 = line_[1]
			#		qnums[job] = path0
			#		jobs.append(job)
			#	dft_subdir[dft_sp_subdir] = jobs
			for i in range(0,Ndump,interval):
				tmp_num += 1
				path0 = dft_sp_subdir + '/' + str(i+1)
				qnums[str(tmp_num)] = path0
				jobs.append(tmp_num)
			dft_subdir[dft_sp_subdir] = jobs

		os.chdir(dft_sp_subdir)
		del job_done2[qnum]
	return qnums, dft_subdir, job_done2

def dft_sp(base_name, base_dir, job_done, interval, dft_input_dir, ncores, queue, walltime, scheduler, modules, python_path='python'):
	os.chdir(base_dir)
	dft_sp_dir = base_dir+'/4_DFT_sp/DFT_sp_'+str(base_name)
	try:
		os.stat(dft_sp_dir)
	except:
		os.makedirs(dft_sp_dir)

	tmp_num = 0
	qnums = {}
	dft_subdir = {}
	jobname = 'DFT_sp'
	job_done2 = copy.deepcopy(job_done)
	for qnum in job_done.keys():
		path = job_done[qnum][0]
		Ndump = job_done[qnum][1]
		cal_name = path.split('/')[-1]
		dft_sp_subdir = dft_sp_dir + '/' + cal_name
		try:
	 		os.stat(dft_sp_subdir)
		except:
			os.mkdir(dft_sp_subdir)
			subprocess.call('cp '+path+'/dynamics_%d.xyz' % Ndump+' '+dft_sp_subdir,shell=True)
			subprocess.call('cp '+dft_input_dir+'/dft_sp.py'+' '+dft_sp_subdir,shell=True)

			job_submit_file(path=dft_sp_subdir,
						    ncores=ncores,
						    walltime=walltime,
						    queue=queue,
						    commands=[python_path+' dft_sp.py'],
						    jobname='DFT_sp',
						    modules=modules,
						    scheduler='slurm')

		os.chdir(dft_sp_subdir)
		jobs = []
		subdirs = []
		if not os.path.exists(dft_sp_subdir+'/submit'):
			xyzf = glob.glob('dynamics_%d.xyz' % Ndump)
			trajectory = read('dynamics_%d.xyz' % Ndump, index=':')
			for i in range(0,Ndump,interval):
				os.chdir(dft_sp_subdir)
				os.mkdir(str(i+1))
				if scheduler == 'slurm':
					subprocess.call('cp dft_sp.py job.slurm '+str(i+1), shell=True)
				elif scheduler == 'pbs':
					subprocess.call('cp dft_sp.py job.pbs '+str(i+1), shell=True)
				os.chdir(dft_sp_subdir+'/'+str(i+1))
				subdirs.append(dft_sp_subdir+'/'+str(i+1))
				trajectory[i].write('POSCAR_1')
				path0 = os.getcwd()
				if scheduler == 'slurm':
					len_qnum0 = 0
					while len_qnum0 == 0:
						p = subprocess.Popen(['sbatch','job.slurm'],stdout=subprocess.PIPE)
						qnum0 = p.communicate()[0].decode("utf-8").split('.')[0]
						len_qnum0 = len(qnum0)
						print(len_qnum0)

					if len_qnum0 != 0:
						qnums[qnum0.split()[-1]] = path0
						jobs.append(qnum0.split()[-1])

				elif scheduler == 'pbs':
					p = subprocess.Popen(['qsub','job.pbs'],stdout=subprocess.PIPE)
					qnum0 = p.communicate()[0].decode("utf-8") # qsub: submit error (Maximum number of jobs already in queue for user MSG=total number of current user's jobs exceeds the queue limit: user yoo    p@halstead-fe00.rcac.purdue.edu, queue standby)
				#deal with submission error
					qjob = qnum0.split('.')[0]
					job_flag = qjob.isdigit()
					while not job_flag:
						p = subprocess.Popen(['qsub','job.pbs'],stdout=subprocess.PIPE)
						qnum0 = p.communicate()[0].decode("utf-8")
						qjob = qnum0.split('.')[0]
						job_flag = qjob.isdigit()

					if len(qnum0) != 0:
						qnums[qnum0.split('.')[0]] = path0
						jobs.append(qnum0.split('.')[0])

			dft_subdir[dft_sp_subdir] = jobs
			with open(dft_sp_subdir+'/submit','w') as submit:
				for job, path in zip(jobs, subdirs):
					submit.write("%s %s\n" % (job, path))
		elif os.path.exists(dft_sp_subdir+'/submit'):
			#with open(dft_sp_subdir+'/submit','r') as submit:
			#	contents = submit.readlines()
			#	for line in contents:
			#		line_ = line.split()
			#		job = line_[0]
			#		path0 = line_[1]
			#		qnums[job] = path0
			#		jobs.append(job)
			#	dft_subdir[dft_sp_subdir] = jobs
			for i in range(0,Ndump,interval):
				tmp_num += 1
				path0 = dft_sp_subdir + '/' + str(i+1)
				qnums[str(tmp_num)] = path0
				jobs.append(tmp_num)
			dft_subdir[dft_sp_subdir] = jobs

		os.chdir(dft_sp_subdir)
		del job_done2[qnum]
	return qnums, dft_subdir, job_done2

def data2images(datafile):
	images = []
	Nimages = 0
	data = open(datafile, 'r')
	data_contents = data.readlines()
	
	for i in range(len(data_contents)):
		line = data_contents[i]
		if line == "begin\n":
			Nimages += 1
			start = i
		
		elif line == "end\n":
			end = i
			singleimage = data_contents[start:end]
			cell_ = singleimage[2:5] 
			atom_lines = singleimage[5:-2]
			energy = float(singleimage[-2].split()[1]) * ( (kcal/mol) / eV)
			cell = []
			for l in cell_:
				cell.append([float(ll) for ll in l.split()[1:]])
			positions = []
			forces = []
			symbols = []
			for atom in atom_lines:
				a = atom.split()
				positions.append([float(p) for p in a[1:4]])
				symbols.append(a[4])
				forces.append([float(f) for f in a[7:]])
			forces = np.array(forces) * ( (kcal/mol) / eV)
			formula = "".join(symbols)
			atoms = Atoms(formula, cell=cell,
						  positions=positions,
						  pbc=[1,1,1])
			atoms.set_calculator(SPC(atoms, energy=energy, forces=forces))
			images.append(atoms)
	data.close()
	return images

def atoms2data(atoms):
	singledata = ""
	energy = atoms.get_potential_energy() * ( eV / (kcal/mol) )
	positions = atoms.get_positions()
	symbols= atoms.get_chemical_symbols()
	forces = atoms.get_forces() * ( eV / (kcal/mol) )
	cell = atoms.get_cell()
	
	singledata += "begin\n"
	singledata += "comment generated by PY\n"
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

	return singledata


def format_dft_data(base_name, base_dir, dft_subdir, dft_job_done, db_dir, eabove=100, db_reduce=False):
	datanames = []
	eabove = eabove * (kcal/mol)/eV #input should be kcal/mol, convert to eV for VASP outcar
	for path in dft_subdir.keys():
		jobs = [str(nn) for nn in dft_subdir[path]]
		all_jobs_done = dft_job_done.keys()
		dataname = db_dir+'/'+ path.split('/')[-1]+'_'+str(base_name)+'.data'
		dataname_original = db_dir+'/'+ path.split('/')[-1]+'_'+str(base_name)+'_original.data'
		#dbname = db_dir+'/'+ path.split('/')[-1]+'_'+str(base_name)+'.db'
		#dbname_original = db_dir+'/'+ path.split('/')[-1]+'_'+str(base_name)+'_original.db'
		print(set(jobs)<=set(all_jobs_done), not os.path.exists(dataname))
		if set(jobs) <= set(all_jobs_done) and not os.path.exists(dataname):
			atoms_list = []
			atoms_string = ""
			for qnum in all_jobs_done:
				if path.split("/")[-1] == dft_job_done[qnum][0].split("/")[-2]:
					images = read(dft_job_done[qnum][0]+'/OUTCAR', index=':')
					if len(images) == 1:
						atoms_list.append(images[0])

						atoms_string += atoms2data(images[0])
					else:
						atoms_list.append(images[0])
						atoms_list.append(images[-1])

						atoms_string += atoms2data(images[0])
						atoms_string += atoms2data(images[-1])

			data_original= open(dataname_original,'w')
			data_original.write(atoms_string)
			data_original.close()
			print("write data_original", dataname_original)


			if db_reduce:
				atoms_list2 = sorted(atoms_list, key=lambda x: x.get_potential_energy())
				e = atoms_list2[0].get_potential_energy() / len(atoms_list2[0])
				atoms_list_reduced = list(filter(lambda x:x.get_potential_energy()/len(x) - e < eabove, atoms_list))
				atoms_string2 = ""
				for atoms in atoms_list_reduced:
					atoms_string2 += atoms2data(atoms)

				data_file =  open(dataname,'w')
				data_file.write(atoms_string2)
				data_file.close()
				print("write data", dataname)
			
			else:
				print('mv '+dataname_original+' '+dataname)
				#subprocess.call('cat '+dataname_original, shell=True)
				subprocess.call('mv '+dataname_original+' '+dataname,shell=True)

				#subprocess.call('cat '+dataname, shell=True)

		datanames.append(dataname)

	return dft_job_done, datanames

def ref_DFT_data(dataname, db_dir, ffield, lmp_path, atomsk_path):
	os.chdir(db_dir)
	images = data2images(dataname)
	new_dataname = dataname[:-5] + '_ref.data'
	new_images = []
	if not os.path.exists(new_dataname):

		s1 = []
		for i in range(len(images)):
			Nelements = len(images[i].get_positions())
			positions = images[i].get_positions()
			energy = images[i].get_potential_energy() * ( eV / (kcal/mol) )
			force = images[i].get_forces()
			cell = images[i].get_cell()
			pbc = [1,1,1]
			cell_string = " ".join([str(c[0])+" "+str(c[1])+" "+str(c[2]) for c in cell])
			cell_string = '"' + cell_string +'"'
			#stress = images[i].stress
			#s1.append([tmp for tmp in stress])
			#stress_string = " ".join([str(-s) for s in stress])
			#stress_string = '"' + stress_string +'"'
			elements = images[i].get_chemical_symbols()
			counter = collections.Counter(elements)
			#print(elements)
			element_short = list(dict.fromkeys(elements))
			element_ordered = list(sorted(element_short,key=lambda e:atomic_numbers[e]))
			print(element_ordered)
			chargefile('charge.txt',element_ordered)
			input_file('base.in',element_ordered, ffield)
			numbers = images[i].get_atomic_numbers()

			ref_ref = 0
			dft_ref = 0

			elements_types = list(set(elements))
			element_ordered = []
			positions_ordered = []
			forces_ordered = []
			for types in elements_types:
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
			formula = ""
			for types in elements_types:
				formula += types + str( element_ordered.count(types) )

			atoms = Atoms(formula, cell=cell, pbc=pbc, positions=positions_ordered)
			chem = atoms.get_chemical_symbols()
			elements = list(set(chem))
			atom_types = {}
			for el, j in zip(elements, range(len(elements))):
				atom_types[el] = j+1

			basename = str(i+1)
			xyz_filename = basename + '.reax'
			write_lammps_data(filename=xyz_filename, atoms=atoms, atom_types=atom_types, units='real')
			atomsk(basename, atomsk_path, xyz_filename)
			lammps(basename, lmp_path, 1)

			#read forces
			ref_force = np.loadtxt('force.dump', skiprows=9)
			dft_force = np.array([list(f) for f in forces_ordered])

			#read energy
			dft_en = energy
			out = open('lmp.out', 'r')
			subprocess.call('cp lmp.out lmp_%d.out' % (i+1), shell=True)
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

			#dft_press = stress
			#DFT - REAX
			en = (dft_en-dft_ref) - (ref_en-ref_ref)
			print(dft_en, dft_ref, ref_en, ref_ref)
			factor = (kcal/mol/eV)
			en = en * factor
			#press = np.array(dft_press) - np.array(ref_press)
			force = [np.array(dft_force[i])*factor-np.array(ref_force[i])*factor for i in range(len(dft_force))]
			new_atoms = Atoms(formula, pbc=pbc,cell=cell,positions=positions_ordered)
			new_atoms.set_calculator(SPC(new_atoms,energy=en,forces=force))
			new_images.append(new_atoms)
		
		string = ""
		for new_image in new_images:
			string += atoms2data(new_image)
		new_data = open(new_dataname,'w')
		new_data.write(string)
		new_data.close()

		subprocess.call('rm lmp_* *.lmp *.reax force.dump lmp.out min.xyz', shell=True)
	return new_dataname


def ref_zbl_DFT_data(dataname, db_dir, lmp_path, atomsk_path):
	os.chdir(db_dir)
	images = data2images(dataname)
	new_dataname = dataname[:-5] + '_ref.data'
	new_images = []
	if not os.path.exists(new_dataname):

		s1 = []
		for i in range(len(images)):
			Nelements = len(images[i].get_positions())
			positions = images[i].get_positions()
			energy = images[i].get_potential_energy() * ( eV / (kcal/mol) )
			force = images[i].get_forces()
			cell = images[i].get_cell()
			pbc = [1,1,1]
			cell_string = " ".join([str(c[0])+" "+str(c[1])+" "+str(c[2]) for c in cell])
			cell_string = '"' + cell_string +'"'
			#stress = images[i].stress
			#s1.append([tmp for tmp in stress])
			#stress_string = " ".join([str(-s) for s in stress])
			#stress_string = '"' + stress_string +'"'
			elements = images[i].get_chemical_symbols()
			counter = collections.Counter(elements)
			#print(elements)
			element_short = list(dict.fromkeys(elements))
			element_ordered = list(sorted(element_short,key=lambda e:atomic_numbers[e]))
			print(element_ordered)
			chargefile('charge.txt',element_ordered)
			zbl_input_file('base.in',element_ordered)
			numbers = images[i].get_atomic_numbers()

			ref_ref = 0
			dft_ref = 0

			elements_types = list(set(elements))
			element_ordered = []
			positions_ordered = []
			forces_ordered = []
			for types in elements_types:
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
			formula = ""
			for types in elements_types:
				formula += types + str( element_ordered.count(types) )

			atoms = Atoms(formula, cell=cell, pbc=pbc, positions=positions_ordered)
			chem = atoms.get_chemical_symbols()
			elements = list(set(chem))
			atom_types = {}
			for el, j in zip(elements, range(len(elements))):
				atom_types[el] = j+1

			basename = str(i+1)
			xyz_filename = basename + '.dat'
			write_lammps_data(filename=xyz_filename, atoms=atoms, atom_types=atom_types, units='real')
			atomsk(basename, atomsk_path, xyz_filename)
			lammps(basename, lmp_path, 1)

			#read forces
			ref_force = np.loadtxt('force.dump', skiprows=9)
			dft_force = np.array([list(f) for f in forces_ordered])

			#read energy
			dft_en = energy
			out = open('lmp.out', 'r')
			subprocess.call('cp lmp.out lmp_%d.out' % (i+1), shell=True)
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

			#dft_press = stress
			#DFT - REAX
			en = (dft_en-dft_ref) - (ref_en-ref_ref)
			print(dft_en, dft_ref, ref_en, ref_ref)
			factor = (kcal/mol/eV)
			en = en * factor
			#press = np.array(dft_press) - np.array(ref_press)
			force = [np.array(dft_force[i])*factor-np.array(ref_force[i])*factor for i in range(len(dft_force))]
			new_atoms = Atoms(formula, pbc=pbc,cell=cell,positions=positions_ordered)
			new_atoms.set_calculator(SPC(new_atoms,energy=en,forces=force))
			new_images.append(new_atoms)
		
		string = ""
		for new_image in new_images:
			string += atoms2data(new_image)
		new_data = open(new_dataname,'w')
		new_data.write(string)
		new_data.close()

		subprocess.call('rm lmp_* *.lmp *.dat force.dump lmp.out min.xyz', shell=True)
	return new_dataname

def ref_script_DFT_data(dataname, db_dir, script, lmp_path, atomsk_path):
	os.chdir(db_dir)
	images = data2images(dataname)
	new_dataname = dataname[:-5] + '_ref.data'
	new_images = []
	if not os.path.exists(new_dataname):

		s1 = []
		for i in range(len(images)):
			Nelements = len(images[i].get_positions())
			positions = images[i].get_positions()
			energy = images[i].get_potential_energy() * ( eV / (kcal/mol) )
			force = images[i].get_forces()
			cell = images[i].get_cell()
			pbc = [1,1,1]
			cell_string = " ".join([str(c[0])+" "+str(c[1])+" "+str(c[2]) for c in cell])
			cell_string = '"' + cell_string +'"'
			#stress = images[i].stress
			#s1.append([tmp for tmp in stress])
			#stress_string = " ".join([str(-s) for s in stress])
			#stress_string = '"' + stress_string +'"'
			elements = images[i].get_chemical_symbols()
			counter = collections.Counter(elements)
			#print(elements)
			element_short = list(dict.fromkeys(elements))
			element_short = ['O','Si','Cu'] #temporary
			element_ordered = list(sorted(element_short,key=lambda e:atomic_numbers[e]))
			chargefile('charge.txt',element_ordered)
			#zbl_input_file('base.in',element_ordered)
			subprocess.call('cp '+script+' ./base.in',shell=True)
			numbers = images[i].get_atomic_numbers()

			ref_ref = 0
			dft_ref = 0

			elements_types = list(set(element_short))
			element_ordered = []
			positions_ordered = []
			forces_ordered = []
			for types in elements_types:
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
			formula = ""
			for types in elements_types:
				formula += types + str( element_ordered.count(types) )

			atoms = Atoms(formula, cell=cell, pbc=pbc, positions=positions_ordered)
			chem = atoms.get_chemical_symbols()
			#elements = list(set(chem))
			atom_types = {}
			for el, j in zip(element_short, range(len(element_short))):
				atom_types[el] = j+1

			basename = str(i+1)
			xyz_filename = basename + '.dat'
			write_lammps_data(filename=xyz_filename, atoms=atoms, atom_types=atom_types, units='real')
			atomsk(basename, atomsk_path, xyz_filename)
			lammps(basename, lmp_path, 1)

			#read forces
			ref_force = np.loadtxt('force.dump', skiprows=9)
			dft_force = np.array([list(f) for f in forces_ordered])

			#read energy
			dft_en = energy
			out = open('lmp.out', 'r')
			subprocess.call('cp lmp.out lmp_%d.out' % (i+1), shell=True)
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

			#dft_press = stress
			#DFT - REAX
			en = (dft_en-dft_ref) - (ref_en-ref_ref)
			print(dft_en, dft_ref, ref_en, ref_ref)
			factor = (kcal/mol/eV)
			en = en * factor
			#press = np.array(dft_press) - np.array(ref_press)
			force = [np.array(dft_force[i])*factor-np.array(ref_force[i])*factor for i in range(len(dft_force))]
			new_atoms = Atoms(formula, pbc=pbc,cell=cell,positions=positions_ordered)
			new_atoms.set_calculator(SPC(new_atoms,energy=en,forces=force))
			new_images.append(new_atoms)
		
		string = ""
		for new_image in new_images:
			string += atoms2data(new_image)
		new_data = open(new_dataname,'w')
		new_data.write(string)
		new_data.close()

		subprocess.call('rm lmp_* *.lmp *.dat force.dump lmp.out min.xyz', shell=True)
	return new_dataname
