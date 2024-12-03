#!/bin/env python
from ase import Atoms, Atom
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.io import read, write
from ase.db import connect
from ase.data import atomic_numbers, atomic_masses
from ase.units import eV, kcal, mol
import numpy as np
import pandas as pd
import sys, os, glob, re, json, time, shutil, collections, subprocess
from shutil import copyfile
from .tracking import job_submit_file

def training_architect(directory, nn_hidden, activation='t'):

	os.chdir(directory)
	new_lines = []
	with open('input.nn','r') as inp:
		lines = inp.readlines()
		for line in lines:
			if 'global_hidden_layers_short' in line:
				line_s = line.split()
				line_s[1] = str(len(nn_hidden))
				line2 = " ".join(line_s) + "\n"
				print(line2)
			elif 'global_nodes_short' in line:
				line_s = line.split()
				line_s = line_s[:1] + [str(h) for h in nn_hidden]
				line2 = " ".join(line_s) + "\n"
				print(line2)
			elif 'global_activation_short' in line:
				line_s = line.split()
				line_s = line_s[:1] + [activation for h in nn_hidden] + ['l']
				line2 = " ".join(line_s) + "\n"
				print(line2)
			else:
				line2 = line
			new_lines.append(line2)

	with open('input.nn','w') as inp2:
		for line in new_lines:
			inp2.write(line)

	return


def use_old_weight(directory):

	os.chdir(directory)
	with open('input.nn','r') as inp:
		lines = inp.readlines()
		check = [True if 'use_old_weights_short' in line else False for line in lines]
		index = check.index(True)
		use_old = lines[index]
		if '#use_' in use_old:
			new_use_old = use_old[1:]
		else:
			new_use_old = use_old
		lines[index] = new_use_old

	with open('input.nn','w') as inp2:
		for line in lines:
			inp2.write(line)

	return


def multi_collect_nnp_data(training_dirs):

	nnp_data_dirs = []
	for a in range(len(training_dirs)):
		training_dir = training_dirs[a]
		nnp_data_dir = collect_nnp_data(training_dir)
		nnp_data_dirs.append(nnp_data_dir)
	return nnp_data_dirs

def multi_ANN_training(base_name,
					   base_dir,
					   db_dir,
					   nnp_input_dir,
					   n2p2_path,
					   modules,
					   architect,
					   use_old=False,
					   ncores=1,
					   queue='standby',
					   walltime='4:00:00',
					   scheduler='slurm'):
	
	training_dirs = []
	qnums_n2p2 = {}
	for a in range(len(architect)):
		subdir ='/NN_%d' % a
		arch = architect[a]
		training_subdir = n2p2_ANN_training(base_name=base_name,
											subdir=subdir,
											base_dir=base_dir,
											db_dir=db_dir,
											nnp_input_dir=nnp_input_dir,
											n2p2_path=n2p2_path,
											modules=modules,
											arch=arch,
											use_old=use_old,
											ncores=ncores,
											queue=queue,
											walltime=walltime,
											scheduler=scheduler)
		training_dirs.append(training_subdir)
		if training_subdir[1] == '0':
			qnums_n2p2[str(a)] = training_subdir[0]
		else:
			qnums_n2p2[training_subdir[1]] = training_subdir[0]

	return qnums_n2p2

def n2p2_ANN_training_nitramine(base_name,
					  base_dir,
					  db_dir,
					  nnp_input_dir, #input for n2p2 training (input.nn)
					  n2p2_path,
					  modules,
					  arch=None,
					  use_old=False,
					  ncores=1,
					  queue='standby',
					  walltime='4:00:00',
					  scheduler='slurm',
					  subdir=''):

	#n2p2 executable commands
	norm = n2p2_path + '/bin/nnp-norm'
	scaling = n2p2_path + '/bin/nnp-scaling 100'
	train = n2p2_path + '/bin/nnp-train'
	predict = n2p2_path + '/bin/nnp-dataset 0'
	qnum = '0'

	os.chdir(base_dir)
	#move to training directory
	training_dir = base_dir + '/2_training/training_' + str(base_name) + subdir

	if os.path.exists(training_dir+'/nnp-data'):
		qnum = '0'
	elif not os.path.exists(training_dir):
	 	#if not os.path.exists(training_dir):
		os.makedirs(training_dir)
		os.chdir(training_dir)

		copyfile(db_dir + '/input.data', training_dir + '/input.data')
		copyfile(nnp_input_dir + '/input.nn', training_dir + '/input.nn')

		prv_weights_path = base_dir+'/2_training/training_'+str(int(base_name)-1)+subdir+'/nnp-data/weights.*.data'
		prv_weights_list = glob.glob(prv_weights_path)
		if use_old == True and len(prv_weights_list) > 0:
			use_old_weight(training_dir)
			subprocess.call('cp '+base_dir+'/2_training/training_'+str(int(base_name)-1)+subdir+'/nnp-data/weights.*.data '+training_dir, shell=True)
			#subprocess.call('cp '+base_dir+'/training_'+str(int(base_name)-1)+subdir+'/nnp-data/weights.*.data '+training_dir, shell=True)

		if arch is not None:
			training_architect(training_dir, arch, activation='t')

		job_submit_file(path=training_dir,
						ncores = ncores,
						walltime=walltime,
						queue=queue,
						commands=['mpiexec -np %s %s\n' % (str(ncores),norm),
						          'cp -f output.nn input.nn\n',
								  'mpiexec -np %s %s\n' % (str(ncores),scaling),
								  'mpiexec -np %s %s\n' % (str(ncores),train)],
						jobname='Training',
						modules=modules,
						scheduler='slurm')

		if scheduler == 'slurm':
			p = subprocess.Popen(['sbatch','job.slurm'],stdout=subprocess.PIPE)
			# .decode("utf-8")
			qnum0 = p.communicate()[0].decode("utf-8").split('.')[0]
			qnum  = qnum0.split()[-1]

		elif scheduler == 'pbs':
			p = subprocess.Popen(['qsub','job.pbs'],stdout=subprocess.PIPE)
			qnum = p.communicate()[0].decode("utf-8").split('.')[0]
			l = 1
			"""
			while l != 0:
				p = subprocess.Popen(['qstat','-f',qnum],stdout=subprocess.PIPE)
				test = p.communicate()[0].decode("utf-8")
				l = len(test.split())
			"""
	os.chdir(base_dir)

	return training_dir, qnum


def n2p2_ANN_training(base_name,
					  base_dir,
					  db_dir,
					  nnp_input_dir, #input for n2p2 training (input.nn)
					  n2p2_path,
					  modules,
					  arch=None,
					  use_old=False,
					  ncores=1,
					  queue='standby',
					  walltime='4:00:00',
					  scheduler='slurm',
					  subdir=''):

	#n2p2 executable commands
	norm = n2p2_path + '/bin/nnp-norm'
	scaling = n2p2_path + '/bin/nnp-scaling 100'
	train = n2p2_path + '/bin/nnp-train'
	predict = n2p2_path + '/bin/nnp-dataset 0'
	qnum = '0'

	os.chdir(base_dir)
	#move to training directory
	training_dir = base_dir + '/2_training/training_' + str(base_name) + subdir
	try:
		os.stat(training_dir)
		if os.path.exists(training_dir+'/nnp-data'):
			qnum = '0'
	except:
		os.makedirs(training_dir)
		os.chdir(training_dir)
		copyfile(db_dir + '/input.data', training_dir + '/input.data')
		copyfile(nnp_input_dir + '/input.nn', training_dir + '/input.nn')
		prv_weights_path = base_dir+'/2_training/training_'+str(int(base_name)-1)+subdir+'/nnp-data/weights.*.data'
		prv_weights_list = glob.glob(prv_weights_path)
		print(len(prv_weights_list))
		if use_old == True and len(prv_weights_list) > 0:
			use_old_weight(training_dir)
			subprocess.call('cp '+base_dir+'/2_training/training_'+str(int(base_name)-1)+subdir+'/nnp-data/weights.*.data '+training_dir, shell=True)

		if arch is not None:
			training_architect(training_dir, arch, activation='t')

		job_submit_file(path=training_dir,
						ncores = ncores,
						walltime=walltime,
						queue=queue,
						commands=['mpiexec -np %s %s\n' % (str(ncores),norm),
						          'cp -f output.nn input.nn\n',
								  'mpiexec -np %s %s\n' % (str(ncores),scaling),
								  'mpiexec -np %s %s\n' % (str(ncores),train)],
						jobname='Training',
						modules=modules,
						scheduler='slurm')

		if scheduler == 'slurm':
			p = subprocess.Popen(['sbatch','job.slurm'],stdout=subprocess.PIPE)
			# .decode("utf-8")
			qnum0 = p.communicate()[0].decode("utf-8").split('.')[0]
			qnum  = qnum0.split()[-1]
			#jobstate = 'PENDING'
			"""
			while jobstate == 'PENDING' or jobstate == 'RUNNING':
				p = subprocess.Popen(['scontrol','show','job',qnum],
						stdout=subprocess.PIPE)
				test = p.communicate()[0].decode("utf-8")
				l = len(test.split())
				jobstate = test.split()[9].split("=")[1]
				print(jobstate)
			"""
		elif scheduler == 'pbs':
			p = subprocess.Popen(['qsub','job.pbs'],stdout=subprocess.PIPE)
			qnum = p.communicate()[0].decode("utf-8").split('.')[0]
			l = 1
			"""
			while l != 0:
				p = subprocess.Popen(['qstat','-f',qnum],stdout=subprocess.PIPE)
				test = p.communicate()[0].decode("utf-8")
				l = len(test.split())
			"""
	os.chdir(base_dir)

	return training_dir, qnum

def collect_nnp_data(training_dir):
	#generate nnp-data for lmp_ann_dynamics
	training_dir0 = training_dir
	os.chdir(training_dir0)
	if os.path.isdir('./nnp-data'):
		weights_list = glob.glob('./nnp-data/weights.*.data')
		input_file = glob.glob('./nnp-data/input.nn')
		scaling_file = glob.glob('./nnp-data/scaling.data')
		if len(weights_list) > 0 and len(input_file) == 1 and len(scaling_file) == 1:
			return training_dir + '/nnp-data'
	else:
		os.mkdir('./nnp-data')
		subprocess.call('cp input.nn ./nnp-data/input.nn',shell=True)
		subprocess.call('cp scaling.data ./nnp-data/scaling.data',shell=True)
		with open('input.nn','r') as f:
			contents = f.readlines()
			for line in contents:
				line2 = line.split()
				if len(line2) > 0 and line2[0] == 'number_of_elements':
					nelements = int(line2[1])
				if len(line2) > 0 and line2[0] == 'elements':
					elements = line2[1:1+nelements]

		for i in range(len(elements)):
			atomic_number = atomic_numbers[elements[i]]
			if atomic_number < 10:
				an = "00" + str(atomic_number)
			elif atomic_number >= 10 and atomic_number < 100:
				an = "0" + str(atomic_number)
			weights = glob.glob('weights.'+an+'*')
			sorted_w = sorted(weights)
			if 'weights.'+an+'.data' in sorted_w:
				sorted_w.remove('weights.'+an+'.data')
			weights2 = sorted_w[-1]
			weights2_data = weights2[:-10] + 'data'
			print(weights2, weights2_data)
			subprocess.call('cp '+weights2+' nnp-data/'+weights2_data,shell=True)
	
		return training_dir + '/nnp-data'

def lmp_ann_dynamics(base_name, base_dir, nnp_data_dir, md_input_dir, lmp_path, ncores, queue, walltime, modules, scheduler='slurm'):
	os.chdir(base_dir)
	md_test_dir = base_dir+'/md_test_'+ str(base_name)
	try:
		os.stat(md_test_dir)
		input_dir = glob.glob(md_input_dir+'/*')
		for test in input_dir:
			md_type = test.split('/')[-1]
			if not os.path.exists(md_type):
				subprocess.call('cp -r '+md_input_dir+'/'+md_type+' '+md_test_dir,shell=True)
		os.stat(base_dir)
	except:
		subprocess.call('cp -r '+md_input_dir+' '+md_test_dir,shell=True)
	dirs = glob.glob(md_test_dir+'/*')

	qnums = {}
	l = 0
	for path in dirs:
		l += 1
		if not os.path.exists(path+'/submit'):
			subprocess.call('cp -r '+nnp_data_dir+' '+path,shell=True)
			os.chdir(path)
		
			job_submit_file(path=path,
							ncores = ncores,
							walltime=walltime,
							queue=queue,
							commands=['mpiexec -np %s %s < nnp.in > out\n' % (str(ncores),lmp_path)],
							jobname='NNRF_dynamics',
							modules=modules,
							scheduler='slurm')

			if scheduler == 'slurm':
				p = subprocess.Popen(['sbatch','job.slurm'],stdout=subprocess.PIPE)
				qnum = p.communicate()[0].decode("utf-8")
				qnums[qnum.split()[-1]] = path

			elif scheduler == 'pbs':
				p = subprocess.Popen(['qsub','job.pbs'],stdout=subprocess.PIPE)
				qnum = p.communicate()[0].decode("utf-8")
				qnums[qnum.split('.')[0]] = path

			with open('submit','w') as submit:
				submit.write(' ')

		elif os.path.exists(path+'/submit'):
			ms = int(time.time()*10000000.0)
			qnums[str(ms+l)] = path

	#job need to track : qnums
	return qnums

def lmp_ann_dynamics_cca(base_name, base_dir, nnp_data_dir, md_input_dir, lmp_path, ncores, queue, walltime, modules, scheduler='slurm'):
	os.chdir(base_dir)
	if not os.path.exists(base_dir+'/3_NNmd'):
		os.makedirs(base_dir+'/3_NNmd')
	md_test_dir = base_dir+'/3_NNmd/md_test_'+ str(base_name)
	try:
		os.stat(md_test_dir)
		input_dir = glob.glob(md_input_dir+'/*')
		for test in input_dir:
			md_type = test.split('/')[-1]
			if not os.path.exists(md_type):
				subprocess.call('cp -r '+md_input_dir+'/'+md_type+' '+md_test_dir,shell=True)
		os.stat(base_dir)
	except:
		subprocess.call('cp -r '+md_input_dir+' '+md_test_dir,shell=True)
	dirs = glob.glob(md_test_dir+'/*')

	qnums = {}
	l = 0

	for path in dirs:
		l += 1
		check_include = path.split('/')[-1]
		if os.path.exists(path+'/iteration'):
			info1 = open(path+'/iteration','r')
			lines = info1.readlines()
			lines1 = [line.strip('\n').split(':') for line in lines]
			for s in lines1:
				if s[0] == 'start':
					start = s[1]
				elif s[0] == 'end':
					end = s[1]
			if int(base_name) < int(start) or int(base_name) > int(end):
				shutil.rmtree(path)
				continue

		elif "from" in check_include and "to" in check_include:
			check_include2 = check_include.split('_')
			start = check_include2[-2][4:]
			end =   check_include2[-1][2:]

			if int(base_name) < int(start) or int(base_name) > int(end):
				shutil.rmtree(path)
				continue

		if not os.path.exists(path+'/submit'):
			subprocess.call('cp -r '+nnp_data_dir+' '+path,shell=True)
			os.chdir(path)

			job_submit_file(path=path,
							ncores = ncores,
							walltime=walltime,
							queue=queue,
							commands=['mpiexec -np %s %s < nnp.in > out\n' % (str(ncores),lmp_path)],
							jobname='NNRF_dynamics',
							modules=modules,
							scheduler='slurm')

			if scheduler == 'slurm':
				p = subprocess.Popen(['sbatch','job.slurm'],stdout=subprocess.PIPE)
				qnum = p.communicate()[0].decode("utf-8")
				qnums[qnum.split()[-1]] = path

			elif scheduler == 'pbs':
				p = subprocess.Popen(['qsub','job.pbs'],stdout=subprocess.PIPE)
				qnum = p.communicate()[0].decode("utf-8")
				qnums[qnum.split('.')[0]] = path

			with open('submit','w') as submit:
				submit.write(' ')
		elif os.path.exists(path+'/submit'):
			ms = int(time.time()*10000000.0)
			qnums[str(ms+l)] = path

	#job need to track : qnums
	return qnums


