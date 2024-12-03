#!/bin/env python
import subprocess 
import numpy as np
#from ase.calculators.lammpslib import write_lammps_data
from ase.calculators.singlepoint import SinglePointCalculator as SPC
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
from .convert import convert_dump_to_xyz_losing, convert_dump_to_xyz_complete

def tracking_qjobs(qnums,types,base_dir,scheduler='slurm'):
	
	#types = 'lmp' or 'vasp'
	job_done = {}
	job_failed = {}
	dft_error = {}

	#Get the current jobids from the cluster in the PBS
	username = 'yoop'
	if scheduler == 'pbs':
		p = subprocess.Popen(['qstat','-u',username],stdout=subprocess.PIPE)
	elif scheduler == 'slurm':
		p = subprocess.Popen(['squeue','-u',username],stdout=subprocess.PIPE)
	qstat = list(p.communicate())[0].decode("utf-8").split("\n")
	qnum_in_queue = []
	for i in range(1,len(qstat)-1):
		qeach = qstat[i].split()
		if (qeach[0].isdigit()):
			qnum_in_queue.append(qeach[0])


	for qnum in qnums.keys():
		if qnum not in qnum_in_queue:
			path = qnums[qnum]

			if types == 'n2p2_training':
				job_done[qnum] = [path, 0]
			
			elif types == 'lmp':
				os.chdir(path)
				dy = glob.glob('dynamics_*.xyz')
				if os.path.exists(path+'/out') and len(dy) == 1:
					Ndump = int( dy[0][:-4].split("_")[1] )
					job_done[qnum] = [path, Ndump]


				elif os.path.exists(path+'/out') and len(dy) == 0:
					with open('out','r') as log:
						contents = log.readlines()
					end = ""
					end_before = ""
					if len(contents) != 0: 
						if len(contents[-2]) != 0 and len(contents[-1]) != 0:
							end_before = contents[-2].split()
							end = contents[-1].split()
					
					if len(end) > 2 and end[0] == 'Total' and end[1] == 'wall':
						#lmp job finished regularly
						for i in range(len(contents)):
							if "Lx Ly Lz" in contents[i]:
								line = contents[i].split()
								Lx_i = line.index('Lx')
								Ly_i = line.index('Ly')
								Lz_i = line.index('Lz')
								Lx = float(contents[i+1].split()[Lx_i])
								Ly = float(contents[i+1].split()[Ly_i])
								Lz = float(contents[i+1].split()[Lz_i])

						dump_filename = path+'/dynamics.xyz'
						dump_filename2 = path+'/dynamics.dump'
						new_dump = len(glob.glob(path+'/dyanmics_*.xyz'))
						Nimages = 0
						if new_dump == 0 and os.path.exists(dump_filename2) and not os.path.exists(dump_filename):
							Nimages = convert_dump_to_xyz_complete(dump_filename2, dump_filename)
							print(Nimages, dump_filename2)
							new_dump_filename = dump_filename[:-4] + '_%d.xyz' % Nimages
							subprocess.call('mv '+dump_filename+' '+new_dump_filename,shell=True)

						if new_dump == 0 and os.path.exists(dump_filename):
							box = [Lx, Ly, Lz]
							dump_xyz = read(dump_filename,index=':')
							Nimages = len(dump_xyz)
							for j in range(len(dump_xyz)):
								dump_xyz[j].set_cell(box)
							write(dump_filename[:-4] + '_%d.xyz' % Nimages ,dump_xyz)
						job_done[qnum] = [path, Nimages]
						
					elif len(end) > 2 and end[0] == 'Last' and end[1] == 'command:':
					#elif len(end) > 2 and end_before[1] == 'Lost' and end_before[2] == 'atoms:':
					#elif len(end) > 2 and end_before[1] == 'Non-numeric' and end_before[2] == 'pressure':
						for i in range(len(contents)):
							if "Lx Ly Lz" in contents[i]:
								line = contents[i].split()
								Lx_i = line.index('Lx')
								Ly_i = line.index('Ly')
								Lz_i = line.index('Lz')
								Lx = float(contents[i+1].split()[Lx_i])
								Ly = float(contents[i+1].split()[Ly_i])
								Lz = float(contents[i+1].split()[Lz_i])


						dump_filename = path+'/dynamics.xyz'
						dump_filename2 = path+'/dynamics.dump'
						new_dump = len(glob.glob(path+'/dyanmics_*.xyz'))
						Nimages = 0
						if new_dump == 0 and os.path.exists(dump_filename2) and not os.path.exists(dump_filename):
							Nimages = convert_dump_to_xyz_losing(dump_filename2, dump_filename)
							new_dump_filename = dump_filename[:-4] + '_%d.xyz' % Nimages
							subprocess.call('mv '+dump_filename+' '+new_dump_filename,shell=True)

						if new_dump == 0 and os.path.exists(dump_filename):
							box = [Lx, Ly, Lz]
							dump_xyz = read(dump_filename,index=':')
							Nimages = len(dump_xyz)
							for j in range(len(dump_xyz)):
								dump_xyz[j].set_cell(box)
							write(dump_filename[:-4] + '_%d.xyz' % Nimages ,dump_xyz)
						job_done[qnum] = [path, Nimages]

					elif len(end_before) > 2 and end_before[0] == 'WARNING:' and end_before[1] == 'H' and end_before[2] == 'matrix':
						for i in range(len(contents)):
							if "Lx Ly Lz" in contents[i]:
								line = contents[i].split()
								Lx_i = line.index('Lx')
								Ly_i = line.index('Ly')
								Lz_i = line.index('Lz')
								Lx = float(contents[i+1].split()[Lx_i])
								Ly = float(contents[i+1].split()[Ly_i])
								Lz = float(contents[i+1].split()[Lz_i])


						dump_filename = path+'/dynamics.xyz'
						dump_filename2 = path+'/dynamics.dump'
						new_dump = len(glob.glob(path+'/dyanmics_*.xyz'))
						Nimages = 0
						if new_dump == 0 and os.path.exists(dump_filename2) and not os.path.exists(dump_filename):
							Nimages = convert_dump_to_xyz_losing(dump_filename2, dump_filename)
							new_dump_filename = dump_filename[:-4] + '_%d.xyz' % Nimages
							subprocess.call('mv '+dump_filename+' '+new_dump_filename,shell=True)

						if new_dump == 0 and os.path.exists(dump_filename):
							box = [Lx, Ly, Lz]
							dump_xyz = read(dump_filename,index=':')
							Nimages = len(dump_xyz)
							for j in range(len(dump_xyz)):
								dump_xyz[j].set_cell(box)
							write(dump_filename[:-4] + '_%d.xyz' % Nimages ,dump_xyz)
						job_done[qnum] = [path, Nimages]

					elif len(end) > 2 and end_before[0] == 'ERROR':
						#dynamics crash losing atoms
						for i in range(len(contents)):
							if "Lx Ly Lz" in contents[i]:
								line = contents[i].split()
								Lx_i = line.index('Lx')
								Ly_i = line.index('Ly')
								Lz_i = line.index('Lz')
								Lx = float(contents[i+1].split()[Lx_i])
								Ly = float(contents[i+1].split()[Ly_i])
								Lz = float(contents[i+1].split()[Lz_i])

						dump_filename = path+'/dynamics.xyz'
						dump_filename2 = path+'/dynamics.dump'
						new_dump = len(glob.glob(path+'/dyanmics_*.xyz'))
						Nimages = 0
						if new_dump == 0 and os.path.exists(dump_filename2) and not os.path.exists(dump_filename):
							Nimages = convert_dump_to_xyz_complete(dump_filename2, dump_filename)
							new_dump_filename = dump_filename[:-4] + '_%d.xyz' % Nimages
							subprocess.call('mv '+dump_filename+' '+new_dump_filename,shell=True)

						if new_dump == 0 and os.path.exists(dump_filename):
							box = [Lx, Ly, Lz]
							dump_xyz = read(dump_filename,index=':')
							Nimages = len(dump_xyz)
							for j in range(len(dump_xyz)):
								dump_xyz[j].set_cell(box)
							write(dump_filename[:-4] + '_%d.xyz' % Nimages ,dump_xyz)
						job_done[qnum] = [path, Nimages]

			elif types == 'vasp':
				os.chdir(path)
				if os.path.exists(path+'/fail'):
					print('dft calculation failure')
					job_done[qnum] = [path, 0]
					job_failed[qnum] = [path, 0]

				elif os.path.exists(path+'/OUTCAR'):
					with open('OUTCAR','r') as vasp_log:
						contents = vasp_log.readlines()
						if len(contents) > 1:
							end = contents[-1].split()
						else: 
							end = [None]
					#print(path, end)

					if len(end) > 1:
						if end[0] == 'Voluntary' and end[1] == 'context':
							#vasp job finished regularly
							job_done[qnum] = [path, 0]
						else:
							dft_error[qnum] = path
					elif len(end) == 1 and end == [None]:
						dft_error[qnum] = path

					elif len(end) == 0:
						dft_error[qnum] = path
						

				else:
					dft_error[qnum] = path
							 	

	return job_done, dft_error, job_failed


def job_submit_file(path,ncores,walltime,queue,commands,jobname,modules,scheduler='slurm'):
	if os.path.exists('/scratch/bell/'):
		ppn = 128
	elif os.path.exists('/scratch/brown/'):
		ppn = 24
	else:
		ppn = 20

	if ncores >= ppn:
		nodes = ncores / ppn
		share = False
	elif ncores < ppn:
		nodes = 1
		share = True
	os.chdir(path)
	if scheduler =='slurm':
		with open(path+'/job.slurm','w') as job:
			job.write('#!/bin/bash\n')
			job.write('#SBATCH -A %s\n' % queue)
			job.write('#SBATCH -N %d\n' % nodes)
			if share:
				job.write('#SBATCH --partition=cluster-shared\n')
				job.write('#SBATCH --ntasks=%d\n' % ncores)
			else:
				job.write('#SBATCH -n %d\n' % ncores)
			job.write('#SBATCH -t %s\n' % walltime)
			job.write('#SBATCH --job-name=%s\n'% jobname)
			for m in modules:
				job.write('module load '+ m +'\n')
			job.write('cd $SLURM_SUBMIT_DIR\n\n')
			for i in range(len(commands)):
				#job.write('mpiexec -np %s %s\n' % (str(ncores),commands[i]))
				job.write('%s\n' % commands[i])
		#p = subprocess.Popen(['sbatch','job.slurm'],stdout=subprocess.PIPE)
		#qnum = p.communicate()[0].decode("utf-8")
		#qnum0 = qnum[-1]
		#path0 = path

	elif scheduler == 'pbs':
		with open(path+'/job.pbs','w') as job:
			job.write('#!/bin/bash\n')
			job.write('#PBS -q %s\n' % queue)
			job.write('#PBS -l nodes=%d:ppn=%d\n' % (nodes,ppn))
			job.write('#PBS -l walltime=%s\n' % time2)
			job.write('#PBS -N %s\n'% jobname)
			job.write('cd $PBS_O_WORKDIR\n\n')
			for m in modules:
				job.write('module load '+ m +'\n')
			for i in range(len(commands)):
				#job.write('mpiexec -np %s %s\n' % (str(ncores),commands[i]))
				job.write('%s\n' % commands[i])

		#p = subprocess.Popen(['qsub','job.pbs'],stdout=subprocess.PIPE)
		#qnum = p.communicate()[0].decode("utf-8")
		#qnum0 = qnum.split('.')[0]
		#apath0 = path
