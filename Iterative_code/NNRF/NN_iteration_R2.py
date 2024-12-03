#!/bin/env python
import subprocess, os, glob, sys, time, json, copy
from .lammpslib import write_lammps_data
# 在NN_iteration_R2.py中
from .n2p2 import (
    n2p2_ANN_training, 
    collect_nnp_data, 
    lmp_ann_dynamics_cca,
    n2p2_ANN_training_nitramine, 
    lmp_ann_dynamics
)  # 移除multi_ANN_training和multi_collect_nnp_data

from .tracking import tracking_qjobs
from .vasp_sp_2 import dft_sp, dft_sp_nitramine, format_dft_data, ref_DFT_data, ref_zbl_DFT_data, ref_script_DFT_data
from .species import species_dft_dynamics, Analyze_database
from .convert import convert_xyz2data, convert_asedb2data

#Dependency
#ase, atomsk, lammps with n2p2

class Iterative_Train:
	############################################################################
	#Periodic system with a good CL+VDW reference potential from reactive force field
	class NNRF_wReaxFFref:
		def __init__(self, 
					 base_dir, 
					 input_db,
					 input_dft,
					 input_md,
					 input_nnrf,
					 commands,
					 env,
					 scheduler='slurm'
					 ):

			if env != None:
				for item in env.items():
					key = item[0]
					value_list = item[1]
					for value in value_list:
						if os.environ.get(key):
							os.environ[key] += os.pathsep + value
						else:
							os.environ[key] = value

			self.base_dir         = base_dir
			self.input_db         = input_db
			self.input_md		  = input_md
			self.input_dft        = input_dft
			self.input_nnrf       = input_nnrf

			self.commands         = commands
			self.n2p2_path        = commands['n2p2']
			self.lmp_path         = commands['lmp']
			self.atomsk_path      = commands['atomsk']
			self.python_path      = commands['python']
			self.scheduler        = scheduler
			self.env              = env


		def Start_DB(self,db_format,reaxff_ref,elements):

			base_dir = self.base_dir
			db_dir   = self.db_dir
			lmp_path 	= self.lmp_path
			atomsk_path = self.atomsk_path

			os.chdir(db_dir)
			dft_dbnames = []

			if db_format == 'ext-xyz':
				ext_xyz_files = glob.glob("*.xyz")
				for xyz_file in ext_xyz_files:
					data_file = convert_xyz2data(xyz_file,xyz_file[:-4]+'.data')
					dft_dbnames.append(data_file)

			self.dft_dbnames =dft_dbnames

			data_files = []
			for dbname in dft_dbnames:
				data_dbname = ref_DFT_data(dataname   = dbname, 
										   db_dir     = db_dir, 
										   element_short = elements,
										   ffield     = reaxff_ref, 
										   lmp_path   = lmp_path, 
										   atomsk_path= atomsk_path)
				data_files.append(data_dbname)

			return data_files

		def Collect_DB(self,iteration,ref_pot=True):

			i = iteration
			base_dir = self.base_dir
			db_dir   = self.input_db
			os.chdir(base_dir)
			base_name = str(i)

			new_db_dir = base_dir+'/1_data/data_'+base_name
			if not os.path.exists(new_db_dir):
				os.makedirs(new_db_dir)
				subprocess.call('cp -r '+db_dir+'/* '+new_db_dir,shell=True)

			subprocess.call('rm '+new_db_dir+'/input.data',shell=True)
			if ref_pot:
				subprocess.call('cat '+new_db_dir+'/*_ref.data > '+new_db_dir+\
													 '/input.data',shell=True)
			else:
				subprocess.call('cat '+new_db_dir+'/*.data > '+new_db_dir+\
													 '/input.data',shell=True)

			self.new_db_dir = new_db_dir
			self.db_dir = db_dir

			return

		def N2P2_train(self,iteration,use_old_weight,training_remote):

			i = iteration
			base_name = str(i)
			base_dir = self.base_dir
			db_dir   = self.new_db_dir
			n2p2_path = self.n2p2_path
			env      = self.env
			training_cores = training_remote['cores']
			training_queue = training_remote['queue']
			training_walltime = training_remote['walltime']
			scheduler = self.scheduler
			training_modules = training_remote['modules']
			nnp_input_dir = self.input_nnrf

			os.chdir(base_dir)

			traindir = base_dir + '/2_training/training_'+base_name
			try:
				os.stat(traindir)
			except:
				os.makedirs(traindir)
				os.chdir(traindir)

			#Submit and Track NN training using n2p2
			training = n2p2_ANN_training_nitramine(base_name=base_name,
											 base_dir     =base_dir,
											 db_dir       =db_dir,
											 nnp_input_dir=nnp_input_dir,
											 n2p2_path    =n2p2_path,
											 ncores       =training_cores,
											 queue        =training_queue,
											 walltime     =training_walltime,
											 use_old      =use_old_weight, # True or False
											 scheduler    =scheduler,
											 modules      =training_modules)
			print(training)
			if training[1] != '0':
				qnum = training[1]
				training_dir = training[0]
			else:
				qnum = '0'
				training_dir = traindir

			qnums_training = {}
			qnums_training[qnum] = training_dir

			submitted_training = sorted(qnums_training.keys())
			flag_train = True
			while flag_train:
				train_job_done, error, job_failed = tracking_qjobs(qnums = qnums_training,
											  types    ='n2p2_training',
											  base_dir =base_dir,
											  scheduler=scheduler)

				finished_training = sorted(train_job_done.keys())
				if submitted_training == finished_training:
					flag_train = False
					training_dir = training_dir
					break

			#Collect the latest parameters from the training directory
			nnp_data_dir = collect_nnp_data(training_dir)
			self.nnp_data_dir = nnp_data_dir

			return

		def LMP_N2P2_dynamics(self,iteration,md_remote):

			i = iteration
			base_name = str(i)
			base_dir = self.base_dir
			db_dir   = self.db_dir
			nnp_data_dir = self.nnp_data_dir
			lmp_path = self.lmp_path
			env      = self.env
			md_cores = md_remote['cores']
			md_queue = md_remote['queue']
			md_walltime = md_remote['walltime']
			scheduler = self.scheduler
			md_modules = md_remote['modules']
			md_input_dir = self.input_md
			os.chdir(base_dir)

			#Submit and Track NN dyanmics using lammps+n2p2
			qnums_lmp = lmp_ann_dynamics_cca(base_name=base_name,
										 base_dir     =base_dir,
										 nnp_data_dir =nnp_data_dir,
										 md_input_dir =md_input_dir,
										 lmp_path     =lmp_path,
										 ncores       =md_cores, 
										 queue        =md_queue,
										 walltime     =md_walltime,
										 scheduler    =scheduler, 
										 modules      =md_modules)

			with open(base_dir+'/log','a') as log:
				log.write("%s iteration ANN dynamic\n" % base_name)
				keys = qnums_lmp.keys()
				for key in keys:
					log.write("%s %s\n" % (key, qnums_lmp[key]))
				log.write("___________________________________\n")

			submitted_lmp = sorted(qnums_lmp.keys())
			flag_lmp = True
			while flag_lmp:
				lmp_job_done, error, job_failed = tracking_qjobs(qnums    =qnums_lmp,
											  types    ='lmp',
											  base_dir =base_dir,
											  scheduler=scheduler)
				finished_lmp = sorted(lmp_job_done.keys())
				if submitted_lmp == finished_lmp:
					flag_lmp = False
					break

			lmp_job_done_dftsp  = copy.deepcopy(lmp_job_done)
			lmp_job_done_species= copy.deepcopy(lmp_job_done)

			self.lmp_job_done_dftsp = lmp_job_done_dftsp
			self.lmp_job_done_species = lmp_job_done_species

			return 

		def vasp_DFT_MDtraj(self,iteration,interval,dft_remote):

			lmp_job_done_dftsp = self.lmp_job_done_dftsp
			i = iteration
			base_name = str(i)
			base_dir = self.base_dir
			db_dir   = self.db_dir
			python_path = self.python_path
			env      = self.env
			dft_cores = dft_remote['cores']
			dft_queue = dft_remote['queue']
			dft_walltime = dft_remote['walltime']
			scheduler = self.scheduler
			dft_modules = dft_remote['modules']
			dft_input_dir = self.input_dft
			os.chdir(base_dir)

			#Submit DFT sp using vasp
			qnums_dft, dft_subdir, job_done = \
			dft_sp(base_name     = base_name,
				   base_dir      = base_dir,
				   job_done      = lmp_job_done_dftsp,
				   interval      = interval,  # need to be controllable
				   dft_input_dir = dft_input_dir,
				   ncores        = dft_cores,
				   queue         = dft_queue,
				   walltime      = dft_walltime,
				   scheduler     = scheduler,
				   modules       = dft_modules,
				   python_path   = python_path)
			
			#Track DFT sp calculations
			with open(base_dir+'/log','a') as log:
				log.write("%s iteration DFT single point\n" % base_name)
				keys = qnums_dft.keys()
				for key in keys:
					log.write("%s %s\n" % (key, qnums_dft[key]))
				log.write("___________________________________\n")

			submitted_dft = sorted(qnums_dft.keys())
			flag_dft = True
			while flag_dft:
				dft_job_done, dft_error, job_failed = tracking_qjobs(qnums=qnums_dft,
											  types     = 'vasp',
											  base_dir  = base_dir,
											  scheduler = scheduler)
				finished_dft  = sorted(dft_job_done.keys())
				if submitted_dft == finished_dft:
					flag_dft = False
					break
				else:
					with open(base_dir+'/dft_errors','w') as d:
						dft_error_path = dft_error.values()
						for each in list(dft_error_path):
							d.write(each+"\n")

			self.dft_subdir = dft_subdir
			self.dft_job_done = dft_job_done

			return

		def vasp_DFT_Species(self,iteration,interval,species_remote,reaxff,ffoption='',ReaxFFBO=True,rdict={}):

			lmp_job_done_species = self.lmp_job_done_species
			i = iteration
			base_name = str(i)
			base_dir = self.base_dir
			db_dir   = self.db_dir
			python_path = self.python_path
			lmp_path 	= self.lmp_path
			atomsk_path = self.atomsk_path 
			env         = self.env
			species_cores = species_remote['cores']
			species_queue = species_remote['queue']
			species_walltime = species_remote['walltime']
			scheduler = self.scheduler
			species_modules = species_remote['modules']
			dft_input_dir = self.input_dft
			ReaxFFBO = ReaxFFBO
			rdict    = rdict

			os.chdir(base_dir)

			exclude = Analyze_database(db_dir+'/input.data')
			qnums_species, species_subdir, job_done_species =\
			species_dft_dynamics(base_name     = base_name,
								 base_dir      = base_dir,
								 dft_input_dir = dft_input_dir,
								 lmp_job_done  = lmp_job_done_species,
								 ncores        = species_cores,
								 queue         = species_queue,
								 walltime      = species_walltime,
								 exclude       = exclude,
								 interval      = interval,
								 ffield_reax   = reaxff,
								 ffoption      = ffoption,
								 lmp_path      = lmp_path,
								 atomsk_path   = atomsk_path,
								 scheduler     = scheduler,
								 modules       = species_modules,
								 rdict         = rdict,
								 ReaxFFBO      = ReaxFFBO, # True or False
								 python_path   = python_path)

			with open(base_dir+'/log','a') as log:
				log.write("%s iteration species DFT dynamics\n" % base_name)
				keys = qnums_species.keys()
				for key in keys:
					log.write("%s %s\n" % (key, qnums_species[key]))
				log.write("___________________________________\n")

			submitted_dft_species = sorted(qnums_species.keys())
			flag_species = True
			while flag_species:
				species_job_done, dft_error, job_failed = tracking_qjobs(qnums=qnums_species,
											  types     = 'vasp',
											  base_dir  = base_dir,
											  scheduler = scheduler)
				finished_dft_species  = sorted(species_job_done.keys())
				if submitted_dft_species == finished_dft_species:
					flag_species = False
					break
				else:
					with open(base_dir+'/dft_species_errors','w') as d:
						dft_error_path = dft_error.values()
						for each in list(dft_error_path):
							d.write(each+"\n")

			self.species_subdir = species_subdir
			self.species_job_done = species_job_done

			return

		def Format_DFT_results(self,iteration,reduce_db):

			i = iteration
			base_name = str(i)
			base_dir = self.base_dir
			db_dir   = self.db_dir
			dft_subdir = self.dft_subdir
			dft_job_done = self.dft_job_done
			species_subdir = self.species_subdir
			species_job_done = self.species_job_done
			python_path = self.python_path
			lmp_path 	= self.lmp_path
			atomsk_path = self.atomsk_path 
			scheduler = self.scheduler
			os.chdir(base_dir)

			if i in reduce_db.keys():
				eabove = reduce_db[i]
				db_reduce = True
			else:
				eabove = 0
				db_reduce = False

			dft_job_done, dft_dbnames = \
			format_dft_data(base_name=base_name,
							 base_dir=base_dir,
							 dft_subdir=dft_subdir,
							 dft_job_done=dft_job_done,
							 db_dir=db_dir,
							 eabove=eabove,
							 db_reduce=db_reduce)

			species_job_done, species_dbnames = \
			format_dft_data(base_name    = base_name,
							base_dir     = base_dir,
							dft_subdir   = species_subdir,
							dft_job_done = species_job_done,
							db_dir       = db_dir,
							eabove       = eabove,
							db_reduce    = False)

			self.dft_dbnames = dft_dbnames
			self.species_dbnames = species_dbnames
			os.chdir(db_dir)

			return

		def DB_ref(self,iteration,reaxff_ref, elements):

			i = iteration
			base_name = str(i)
			base_dir = self.base_dir
			db_dir   = self.db_dir
			dft_dbnames = self.dft_dbnames
			species_dbnames = self.species_dbnames
			lmp_path 	= self.lmp_path
			atomsk_path = self.atomsk_path
			env         = self.env

			data_files = []
			for dbname in dft_dbnames:
				data_dbname = ref_DFT_data(dataname   = dbname, 
										   db_dir     = db_dir, 
										   element_short = elements,
										   ffield     = reaxff_ref, 
										   lmp_path   = lmp_path, 
										   atomsk_path= atomsk_path)
				data_files.append(data_dbname)

			for dbname2 in species_dbnames:
				data_dbname = ref_DFT_data(dataname   = dbname2, 
										   db_dir     = db_dir, 
										   element_short = elements,
										   ffield     = reaxff_ref, 
										   lmp_path   = lmp_path, 
										   atomsk_path= atomsk_path)
				data_files.append(data_dbname)

			return data_files

	##############################

	###Periodic system without a good reference potential
	class NNRF_wRef:
		def __init__(self,
					 base_dir,
					 input_db,
					 input_dft,
					 input_md,
					 input_nnrf,
					 NN_architect,
					 commands,
					 env,
					 scheduler='slurm'
					 ):


			if env != None:
				for item in env.items():
					key = item[0]
					value_list = item[1]
					for value in value_list:
						if os.environ.get(key):
							os.environ[key] += os.pathsep + value
						else:
							os.environ[key] = value

			self.base_dir         = base_dir
			self.input_db         = input_db
			self.input_md         = input_md
			self.input_dft        = input_dft
			self.input_nnrf       = input_nnrf

			self.NN_architect     = NN_architect
			self.commands         = commands
			self.n2p2_path        = commands['n2p2']
			self.lmp_path         = commands['lmp']
			self.atomsk_path      = commands['atomsk']
			self.python_path      = commands['python']
			self.scheduler        = scheduler
			self.env              = env

		def Start_DB(self,db_format,ref_script,elements):

			base_dir = self.base_dir
			db_dir   = self.input_db
			lmp_path    = self.lmp_path
			atomsk_path = self.atomsk_path

			os.chdir(db_dir)
			dft_dbnames = []
			if db_format == 'ext-xyz':
				ext_xyz_files = glob.glob("*.xyz")
				for xyz_file in ext_xyz_files:
					data_file = convert_xyz2data(xyz_file,xyz_file[:-4]+'.data')
					dft_dbnames.append(data_file)
			elif db_format == 'ase-db':
				asedb_files = glob.glob("*.db")
				for db_file in asedb_files:
					data_file = convert_asedb2data(db_file,db_file[:-3]+'.data')
					dft_dbnames.append(data_file)
				
			self.dft_dbnames = dft_dbnames

			ref_script     = copy.deepcopy(ref_script)
			if ref_script != None:
				for k, path in ref_script.items():
					ref_script[k] = path

				data_files = []
				for dbname in dft_dbnames:
					data_dbname = ref_script_DFT_data(dataname   = dbname, 
											          db_dir     = db_dir, 
											          script     = ref_script,
													  element_short = elements,
											          lmp_path   = lmp_path, 
											          atomsk_path= atomsk_path)
					data_files.append(data_dbname)
			else:
				data_files = dft_dbnames

			return data_files

		def Collect_DB(self,iteration,ref_pot=True):

			i = iteration
			base_name = str(i)
			base_dir = self.base_dir
			db_dir   = self.input_db
			os.chdir(base_dir)

			new_db_dir = base_dir +'/1_data/data_'+base_name
			if not os.path.exists(new_db_dir):
				os.makedirs(new_db_dir)
				subprocess.call('cp -r '+db_dir+'/* '+new_db_dir,shell=True)
			subprocess.call('rm '+new_db_dir+'/input.data',shell=True)
			if ref_pot:
				subprocess.call('cat '+new_db_dir+'/*_ref.data > '+new_db_dir+\
													 '/input.data',shell=True)
			else:
				subprocess.call('cat '+new_db_dir+'/*.data > '+new_db_dir+\
													 '/input.data',shell=True)

			self.new_db_dir = new_db_dir
			self.db_dir     = db_dir

			return

		def N2P2_train(self,iteration,use_old_weight,training_remote):

			i = iteration
			base_name = str(i)
			base_dir  = self.base_dir
			db_dir    = self.new_db_dir
			n2p2_path = self.n2p2_path
			env       = self.env
			training_cores = training_remote['cores']
			training_queue = training_remote['queue']
			training_walltime = training_remote['walltime']
			NN_architect = self.NN_architect
			scheduler = self.scheduler
			training_modules = training_remote['modules']
			nnp_input_dir = self.input_nnrf
			
			traindir = base_dir + '/2_training/training_'+base_name
			try:
				os.stat(traindir)
			except:
				os.makedirs(traindir)
				os.chdir(traindir)

			#Submit multiple NN training using n2p2
			qnums_training  = multi_ANN_training(base_name		 = base_name,
											   base_dir		 = base_dir,
											   db_dir		 = db_dir,
											   nnp_input_dir = nnp_input_dir,
											   n2p2_path	 = n2p2_path,
											   modules		 = training_modules,
											   architect	 = NN_architect,
											   use_old		 = use_old_weight, #True or False
											   ncores		 = training_cores,
											   queue		 = training_queue,
											   walltime		 = training_walltime,
											   scheduler	 = scheduler)
			
			#Tracking the multiple NNs
			submitted_training = sorted(qnums_training.keys())
			flag_train = True
			while flag_train:
				train_job_done, error, job_failed = tracking_qjobs(qnums=qnums_training,
							   types	= 'n2p2_training',
							   base_dir	= base_dir,
							   scheduler= scheduler)
				finished_training = sorted(train_job_done.keys())
				if submitted_training == finished_training:
					flag_train = False
					break

			training_dirs = list(qnums_training.values())
			# Write a file indicating the job is done
			nnp_data_dirs = multi_collect_nnp_data(training_dirs)
			self.nnp_data_dirs = nnp_data_dirs

			return


		def LMP_N2P2_dynamics(self,iteration,md_remote):

			i = iteration
			base_name = str(i)
			base_dir  = self.base_dir
			db_dir    = self.db_dir
			nnp_data_dirs = self.nnp_data_dirs
			lmp_path  = self.lmp_path
			env       = self.env
			md_cores = md_remote['cores']
			md_queue = md_remote['queue']
			md_walltime = md_remote['walltime']
			md_modules = md_remote['modules']
			scheduler = self.scheduler
			md_input_dir = self.input_md
			os.chdir(base_dir)

			#Use one force field: Primary NN 
			Primary_nnp_data = nnp_data_dirs[0]

			#Submit and Track NN dyanmics using lammps+n2p2
			qnums_lmp = lmp_ann_dynamics_cca(base_name		= base_name,
											 base_dir 		= base_dir,
											 nnp_data_dir	= Primary_nnp_data,
											 md_input_dir	= md_input_dir,
											 lmp_path 		= lmp_path,
											 ncores 		= md_cores,
											 queue 			= md_queue,
											 walltime 		= md_walltime,
											 scheduler 		= scheduler,
											 modules 		= md_modules)

			with open(base_dir+'/log','a') as log:
				log.write("%s iteration ANN dynamic\n" % base_name)
				keys = qnums_lmp.keys()
				for key in keys:
					log.write("%s %s\n" % (key, qnums_lmp[key]))
				log.write("___________________________________\n")

			submitted_lmp = sorted(qnums_lmp.keys())
			flag_lmp = True
			while flag_lmp:
				lmp_job_done, error, job_failed = tracking_qjobs(qnums    =qnums_lmp,
											  types    ='lmp',
											  base_dir =base_dir,
											  scheduler=scheduler)
				finished_lmp = sorted(lmp_job_done.keys())
				if submitted_lmp == finished_lmp:
					flag_lmp = False
					break

			lmp_job_done_dftsp  = copy.deepcopy(lmp_job_done)
			self.lmp_job_done_dftsp = lmp_job_done_dftsp

			return

		def vasp_DFT_MDtraj(self,iteration,interval,dft_remote):

			lmp_job_done_dftsp = self.lmp_job_done_dftsp
			i = iteration
			base_name = str(i)
			base_dir  = self.base_dir
			db_dir    = self.db_dir
			python_path = self.python_path
			env         = self.env
			dft_cores = dft_remote['cores']
			dft_queue = dft_remote['queue']
			dft_walltime = dft_remote['walltime']
			dft_modules = dft_remote['modules']
			scheduler = self.scheduler
			dft_input_dir = self.input_dft
			os.chdir(base_dir)

			#Submit DFT sp using vasp
			qnums_dft, dft_subdir, job_done = \
			dft_sp(base_name     = base_name,
				   base_dir      = base_dir,
				   job_done      = lmp_job_done_dftsp,
				   interval      = interval,
				   dft_input_dir = dft_input_dir,
				   ncores        = dft_cores,
				   queue         = dft_queue,
				   walltime      = dft_walltime,
				   scheduler     = scheduler,
				   modules       = dft_modules,
				   python_path   = python_path)

			submitted_dft = sorted(qnums_dft.keys())
			flag_dft = True
			while flag_dft:
				dft_job_done, dft_error, job_failed = tracking_qjobs(qnums=qnums_dft,
											  types     = 'vasp',
											  base_dir  = base_dir,
											  scheduler = scheduler)
				finished_dft  = sorted(dft_job_done.keys())
				print(len(submitted_dft), len(finished_dft))
				if submitted_dft == finished_dft:
					flag_dft = False
					break
				else:
					with open(base_dir+'/dft_errors','w') as d:
						dft_error_path = dft_error.values()
						for each in list(dft_error_path):
							d.write(each+"\n")

			#Track DFT sp calculations
			with open(base_dir+'/log','a') as log:
				log.write("%s iteration DFT single point\n" % base_name)
				keys = qnums_dft.keys()
				for key in keys:
					log.write("%s %s\n" % (key, qnums_dft[key]))
				log.write("___________________________________\n")

			#print(job_failed)
			if len(job_failed.values()) > 0:
				for key in job_failed.keys():
					v = job_failed[key]
					dft_job_done = {key:val for key, val in dft_job_done.items() if val != v}
					st = v[0].split("/")[:-1]
					v2 = "/".join(st)
					dft_subdir[v2].remove(int(key))

			self.dft_subdir = dft_subdir
			self.dft_job_done = dft_job_done


			return


		def Format_DFT_results(self,iteration,reduce_db):

			i = iteration
			base_name = str(i)
			base_dir  = self.base_dir
			db_dir    = self.db_dir
			dft_subdir = self.dft_subdir
			dft_job_done = self.dft_job_done
			python_path = self.python_path
			lmp_path    = self.lmp_path
			atomsk_path = self.atomsk_path
			env         = self.env
			scheduler = self.scheduler
			os.chdir(base_dir)

			if i in reduce_db.keys():
				eabove = reduce_db[i]
				db_reduce = True
			else:
				eabove = 0
				db_reduce = False

			dft_job_done, dft_dbnames = \
			format_dft_data(base_name		= base_name,
							 base_dir		= base_dir,
							 dft_subdir		= dft_subdir,
							 dft_job_done	= dft_job_done,
							 db_dir			= db_dir,
							 eabove			= eabove,
							 db_reduce		= db_reduce)

			self.dft_dbnames = dft_dbnames
			os.chdir(db_dir)
			
			return


		def DB_ref(self,iteration,ref_script,elements):

			i = iteration
			base_name = str(i)
			base_dir = self.base_dir
			db_dir   = self.db_dir
			dft_dbnames = self.dft_dbnames
			lmp_path    = self.lmp_path
			atomsk_path = self.atomsk_path
			env         = self.env

			ref_script     = copy.deepcopy(ref_script)
			if ref_script != None:
				for k, path in ref_script.items():
					ref_script[k] = path
					
				data_files = []
				for dbname in dft_dbnames:
					data_dbname = ref_script_DFT_data(dataname   = dbname, 
											   db_dir     = db_dir, 
											   script     = ref_script,
											   element_short = elements,
											   lmp_path   = lmp_path, 
											   atomsk_path= atomsk_path)
					data_files.append(data_dbname)

			else:
				data_files = dft_dbnames

			return

