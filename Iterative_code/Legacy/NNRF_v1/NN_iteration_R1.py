#!/bin/env python
import subprocess, os, glob, sys, time, json, copy
from .lammpslib import write_lammps_data
from .n2p2 import n2p2_ANN_training, collect_nnp_data, lmp_ann_dynamics_cca, multi_ANN_training, multi_collect_nnp_data,n2p2_ANN_training_nitramine, lmp_ann_dynamics
from .tracking import tracking_qjobs
from .vasp_sp_2 import dft_sp, dft_sp_nitramine, format_dft_data, ref_DFT_data, ref_zbl_DFT_data, ref_script_DFT_data
from .species import species_dft_dynamics, Analyze_database

#Dependency
#ase, atomsk, lammps with n2p2

class Iterative_Train:

	############################################################################
	#Periodic system with a good CL+VDW reference potential from reactive force field

	class NNRF_wReaxFFref:
		def __init__(self, 
					 base_dir, 
					 ffield, 
					 ffield_reax,
					 input_dirs,
					 training_remote,
					 md_remote,
					 dft_remote,
					 commands,
					 scheduler='slurm'):

			self.scheduler        = scheduler
			self.base_dir         = base_dir

			self.db_dir           = base_dir + "/" + input_dirs['db_dir']
			self.md_input_dir     = base_dir + "/" + input_dirs['md_input_dir']
			self.dft_input_dir    = base_dir + "/" + input_dirs['dft_input_dir']
			self.nnp_input_dir    = base_dir + "/" + input_dirs['nnp_input_dir']

			self.ffield           = base_dir + "/" + ffield
			self.ffield_reax      = base_dir + "/" + ffield_reax

			self.training_cores   = training_remote['training_cores']
			self.training_queue   = training_remote['training_queue']
			self.training_walltime= training_remote['training_walltime']
			self.training_modules = training_remote['training_modules']

			self.md_cores         = md_remote['md_cores']
			self.md_queue         = md_remote['md_queue']
			self.md_walltime      = md_remote['md_walltime']
			self.md_modules       = md_remote['md_modules']

			self.dft_cores        = dft_remote['dft_cores']
			self.dft_queue        = dft_remote['dft_queue']
			self.dft_walltime     = dft_remote['dft_walltime']
			self.dft_interval     = dft_remote['dft_interval']
			self.species_cores    = dft_remote['species_cores']
			self.species_queue    = dft_remote['species_queue']
			self.species_walltime = dft_remote['species_walltime']
			self.rdict            = dft_remote['rdict']
			self.ReaxFFBO         = dft_remote['ReaxFFBO']
			self.dft_modules      = dft_remote['dft_modules']

			self.commands         = commands
			self.scheduler        = scheduler

			self.n2p2_path        = commands['n2p2']
			self.lmp_path         = commands['lmp']
			self.atomsk_path      = commands['atomsk']
			self.python_path      = commands['python']

		def Iteration(self, 
					  use_old_weight=5, 
				      reduce_db={}, 
					  iteration=20, 
					  restart=1, 
					  start_species=20):

			self.start_species    = start_species
			self.reduce_db        = reduce_db
			self.iteration        = iteration
			self.restart          = restart
			self.use_old_weight   = use_old_weight

			base_dir       = self.base_dir
			db_dir         = self.db_dir
			md_input_dir   = self.md_input_dir
			nnp_input_dir  = self.nnp_input_dir
			dft_input_dir  = self.dft_input_dir

			n2p2_path      = self.n2p2_path
			lmp_path       = self.lmp_path
			atomsk_path    = self.atomsk_path
			python_path    = self.python_path

			training_cores = self.training_cores
			training_queue = self.training_queue
			training_walltime=self.training_walltime
			training_modules =self.training_modules

			md_cores       = self.md_cores
			md_queue       = self.md_queue         
			md_walltime    = self.md_walltime      
			md_modules     = self.md_modules

			dft_cores      = self.dft_cores
			dft_queue      = self.dft_queue        
			dft_walltime   = self.dft_walltime     
			dft_interval   = self.dft_interval
			species_cores  = self.species_cores    
			species_queue  = self.species_queue    
			species_walltime=self.species_walltime 
			dft_modules    = self.dft_modules

			scheduler      = self.scheduler
			ffield         = self.ffield
			ffield_reax    = self.ffield_reax
			ReaxFFBO       = self.ReaxFFBO
			rdict          = self.rdict

			with open(base_dir+'/log','w') as log:
				log.write('start the training iterations\n')

			for i in range(restart, iteration+1):
				base_name = str(i)

				if use_old_weight <= i:
					use_old = True
				else:
					use_old = False
				#print("use_old", use_old_weight, i, use_old)
				os.chdir(base_dir)

				new_db_dir = base_dir+'/1_data/data_'+base_name
				if not os.path.exists(new_db_dir):
					os.makedirs(new_db_dir)
					subprocess.call('cp -r '+db_dir+'/* '+new_db_dir,shell=True)

				subprocess.call('rm '+new_db_dir+'/input.data',shell=True)
				subprocess.call('cat '+new_db_dir+'/*_ref.data > '+new_db_dir+\
													 '/input.data',shell=True)
				traindir = base_dir + '/2_training/training_'+base_name
				try:
					os.stat(traindir)
				except:
					os.makedirs(traindir)
					os.chdir(traindir)

				#Submit and Track NN training using n2p2
				training = n2p2_ANN_training_nitramine(base_name=base_name,
												 base_dir     =base_dir,
												 db_dir       =new_db_dir,
												 nnp_input_dir=nnp_input_dir,
												 n2p2_path    =n2p2_path,
												 ncores       =training_cores,
												 queue        =training_queue,
												 walltime     =training_walltime,
												 use_old      =use_old,
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
				print(nnp_data_dir)

				#Submit and Track NN dyanmics using lammps+n2p2
				qnums_lmp = lmp_ann_dynamics_cca(base_name    =base_name,
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

				#Submit DFT sp using vasp
				qnums_dft, dft_subdir, job_done = \
				dft_sp(base_name     = base_name,
					   base_dir      = base_dir,
					   job_done      = lmp_job_done_dftsp,
					   interval      = dft_interval,  # need to be controllable
					   dft_input_dir = dft_input_dir,
					   ncores        = dft_cores,
					   queue         = dft_queue,
					   walltime      = dft_walltime,
					   scheduler     = scheduler,
					   modules       = dft_modules,
					   python_path    = python_path)
				
				#Species analysis before starting to check DFT sp calculations
				if int(base_name) >= self.start_species:
					exclude = Analyze_database(new_db_dir+'/input.data')
					qnums_species, species_subdir, job_done_species =\
					species_dft_dynamics(base_name     = base_name,
										 base_dir      = base_dir,
										 dft_input_dir = dft_input_dir,
										 lmp_job_done  = lmp_job_done_species,
										 ncores        = species_cores,
										 queue         = species_queue,
										 walltime      = species_walltime,
										 exclude       = exclude,
				 						 interval      = dft_interval,
										 ffield_reax   = ffield_reax,
										 ffoption      = '',
										 lmp_path      = lmp_path,
										 atomsk_path   = atomsk_path,
										 scheduler     = scheduler,
										 modules       = dft_modules,
										 rdict         = rdict,
										 ReaxFFBO      = ReaxFFBO,
										 python_path   = python_path)

					with open(base_dir+'/log','a') as log:
						log.write("%s iteration species DFT dynamics\n" % base_name)
						keys = qnums_species.keys()
						for key in keys:
							log.write("%s %s\n" % (key, qnums_species[key]))
						log.write("___________________________________\n")

					submitted_dft_species = sorted(qnums_species.keys())

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


				if i in reduce_db.keys():
					eabove = reduce_db[i]
					db_reduce = True
				else:
				 	eabove = 0
				 	db_reduce = False

				dft_job_done, dbnames = \
				format_dft_data(base_name=base_name,
						         base_dir=base_dir,
								 dft_subdir=dft_subdir,
								 dft_job_done=dft_job_done,
								 db_dir=db_dir,
								 eabove=eabove,
								 db_reduce=db_reduce)

				if int(base_name) >= self.start_species:
					flag_species = True
					while flag_species:
						dft_job_done2, error, job_failed = tracking_qjobs(qnums    = qnums_species,
													   types    = 'vasp',
													   base_dir = base_dir,
													   scheduler= scheduler)
						finished_dft_species  = sorted(dft_job_done2.keys())

						print(len(submitted_dft_species), len(finished_dft_species))

						if submitted_dft_species == finished_dft_species:
							flag_dft_species = False
							break

					dft_job_done_, species_dbnames = \
					format_dft_data(base_name    = base_name,
							        base_dir     = base_dir,
									dft_subdir   = species_subdir,
									dft_job_done = dft_job_done2,
									db_dir       = db_dir,
									eabove       = eabove,
									db_reduce    = False)
	
				data_files = []
				for dbname in dbnames:
					data_dbname = ref_DFT_data(dataname   = dbname, 
							                   db_dir     = db_dir, 
											   ffield     = ffield, 
											   lmp_path   = lmp_path, 
											   atomsk_path= atomsk_path)
					data_files.append(data_dbname)
				if int(base_name) >= self.start_species:
					for dbname2 in species_dbnames:
						data_dbname = ref_DFT_data(dataname   = dbname, 
												   db_dir     = db_dir, 
												   ffield     = ffield, 
												   lmp_path   = lmp_path, 
												   atomsk_path= atomsk_path)
						data_files.append(data_dbname)
				os.chdir(db_dir)


			return

	###Periodic system without a good reference potential
	class NNRF_wRef:
		def __init__(self,
					base_dir,
					input_dirs,
					training_remote,
					md_remote,
					dft_remote,
					commands, 
					ref_script=None,
					scheduler='slurm'):

			self.scheduler        = scheduler
			self.base_dir         = base_dir

			self.db_dir           = base_dir + "/" + input_dirs['db_dir']
			self.md_input_dir     = base_dir + "/" + input_dirs['md_input_dir']
			self.dft_input_dir    = base_dir + "/" + input_dirs['dft_input_dir']
			self.nnp_input_dir    = base_dir + "/" + input_dirs['nnp_input_dir']

			self.training_cores   = training_remote['training_cores']
			self.training_queue   = training_remote['training_queue']
			self.training_walltime= training_remote['training_walltime']
			self.training_modules = training_remote['training_modules']

			self.md_cores         = md_remote['md_cores']
			self.md_queue         = md_remote['md_queue']
			self.md_walltime      = md_remote['md_walltime']
			self.md_modules       = md_remote['md_modules']

			self.dft_cores        = dft_remote['dft_cores']
			self.dft_queue        = dft_remote['dft_queue']
			self.dft_walltime     = dft_remote['dft_walltime']
			self.dft_modules      = dft_remote['dft_modules']
			self.dft_interval     = dft_remote['dft_interval']

			self.commands         = commands

			self.n2p2_path        = commands['n2p2']
			self.lmp_path         = commands['lmp']
			self.atomsk_path      = commands['atomsk']
			self.python_path      = commands['python']

			self.ref_script       = ref_script

		def Iteration(self,
				      NN_architect=[[20,20]], 
					  use_old_weight=10,
					  reduce_db={},
				      iteration=20, 
					  restart=1):

			self.NN_architect     = NN_architect
			self.use_old_weight   = use_old_weight
			self.iteration        = iteration
			self.restart          = restart

			base_dir       = self.base_dir
			db_dir         = self.db_dir
			md_input_dir   = self.md_input_dir
			nnp_input_dir  = self.nnp_input_dir
			dft_input_dir  = self.dft_input_dir

			n2p2_path      = self.n2p2_path
			lmp_path       = self.lmp_path
			atomsk_path    = self.atomsk_path
			python_path    = self.python_path

			training_cores = self.training_cores
			training_queue = self.training_queue
			training_walltime=self.training_walltime
			training_modules =self.training_modules

			md_cores       = self.md_cores
			md_queue       = self.md_queue         
			md_walltime    = self.md_walltime      
			md_modules     = self.md_modules

			dft_cores      = self.dft_cores
			dft_queue      = self.dft_queue        
			dft_walltime   = self.dft_walltime  
			dft_modules    = self.dft_modules
			dft_interval   = self.dft_interval

			ref_script     = self.ref_script

			scheduler      = self.scheduler

			with open(base_dir+'/log','w') as log:
				log.write('start the training iterations\n')

			for i in range(restart, iteration+1):
				base_name = str(i)				

				if use_old_weight <= i:
					use_old = True
				else:
					use_old = False
				print("use_old", use_old_weight, i, use_old)
				os.chdir(base_dir)

				new_db_dir = base_dir +'/1_data/data_'+base_name
				if not os.path.exists(new_db_dir):
					os.makedirs(new_db_dir)
					subprocess.call('cp -r '+db_dir+'/* '+new_db_dir,shell=True)
				subprocess.call('rm '+new_db_dir+'/input.data',shell=True)
				if ref_script != None:
					subprocess.call('cat '+new_db_dir+'/*_ref.data > '+new_db_dir+\
														 '/input.data',shell=True)
				else:
					subprocess.call('cat '+new_db_dir+'/*.data > '+new_db_dir+\
														 '/input.data',shell=True)

				traindir = base_dir + '/2_training/training_'+base_name
				try:
					os.stat(traindir)
				except:
					os.makedirs(traindir)
					os.chdir(traindir)

				#Submit multiple NN training using n2p2
				qnums_n2p2  = multi_ANN_training(base_name		 = base_name,
												   base_dir		 = base_dir,
												   db_dir		 = new_db_dir,
												   nnp_input_dir = nnp_input_dir,
												   n2p2_path	 = n2p2_path,
												   modules		 = training_modules,
												   architect	 = NN_architect,
												   use_old		 = use_old,
												   ncores		 = training_cores,
												   queue		 = training_queue,
												   walltime		 = training_walltime,
												   scheduler	 = scheduler)
				
				#Tracking the multiple NNs
				submitted_n2p2 = sorted(qnums_n2p2.keys())
				flag_n2p2 = True
				while flag_n2p2:
					n2p2_job_done, error, job_failed = tracking_qjobs(qnums=qnums_n2p2,
					               types	= 'n2p2_training',
								   base_dir	= base_dir,
								   scheduler= scheduler)
					finished_n2p2 = sorted(n2p2_job_done.keys())
					if submitted_n2p2 == finished_n2p2:
						flag_n2p2 = False
						break

				training_dirs = list(qnums_n2p2.values())
				# Write a file indicating the job is done

				nnp_data_dirs = multi_collect_nnp_data(training_dirs)

				#Use one force field: Primary NN 
				Primary_nnp_data = nnp_data_dirs[0]
				#print(nnp_data_dirs)
				#print('Prime',Primary_nnp_data)
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

				#Start the multiple-fit procedure
				
				#Run NNP single point
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

				#Submit DFT sp using vasp
				qnums_dft, dft_subdir, job_done = \
				dft_sp(base_name     = base_name,
					   base_dir      = base_dir,
					   job_done      = lmp_job_done_dftsp,
					   interval      = dft_interval,
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

				if i in reduce_db.keys():
					eabove = reduce_db[i]
					db_reduce = True
				else:
				 	eabove = 0
				 	db_reduce = False

				dft_job_done, dbnames = \
				format_dft_data(base_name		= base_name,
						         base_dir		= base_dir,
								 dft_subdir		= dft_subdir,
								 dft_job_done	= dft_job_done,
								 db_dir			= db_dir,
								 eabove			= eabove,
								 db_reduce		= db_reduce)

				if ref_script != None:
					data_files = []
					for dbname in dbnames:
						data_dbname = ref_script_DFT_data(dataname   = dbname, 
												   db_dir     = db_dir, 
												   script     = base_dir+'/'+ref_script,
												   lmp_path   = lmp_path, 
												   atomsk_path= atomsk_path)
						data_files.append(data_dbname)

				else:
					data_files = dbnames

				os.chdir(db_dir)


			return
	
	###Periodic system without a good reference potential
	class ComplexCompounAlloy:
		def __init__(self,
					base_dir, db_dir, md_input_dir,
					dft_input_dir, nnp_input_dir,
					training_cores, training_queue, training_walltime,
					md_cores,       md_queue,       md_walltime,
					dft_cores,      dft_queue,      dft_walltime, dft_interval,
					commands, scheduler='slurm',
					modules=['intel/17.0.1.132 impi/2017.1.132']):

			self.scheduler        = scheduler
			self.modules          = modules

			self.base_dir         = base_dir
			self.db_dir           = base_dir + "/" + db_dir
			self.md_input_dir     = base_dir + "/" + md_input_dir
			self.dft_input_dir    = base_dir + "/" + dft_input_dir
			self.nnp_input_dir    = base_dir + "/" + nnp_input_dir

			self.training_cores   = training_cores
			self.training_queue   = training_queue
			self.training_walltime= training_walltime
			self.md_cores         = md_cores
			self.md_queue         = md_queue
			self.md_walltime      = md_walltime
			self.dft_cores        = dft_cores
			self.dft_queue        = dft_queue
			self.dft_walltime     = dft_walltime
			self.dft_interval     = dft_interval
			self.commands         = commands

			self.n2p2_path        = commands['n2p2']
			self.lmp_path         = commands['lmp']
			self.atomsk_path      = commands['atomsk']

		def Iteration(self,
				      NN_architect=[[10,10,10],[20,20]], 
				      iteration=20, 
					  restart=1,
					  use_old_weight=10,
					  mf_process=20,
					  reduce_db={},
					  ref_zbl=True,
					  ref_script=None):

			self.NN_architect     = NN_architect
			self.iteration        = iteration
			self.restart          = restart
			self.mf_process       = mf_process
			self.use_old_weight   = use_old_weight

			base_dir       = self.base_dir
			db_dir         = self.db_dir
			md_input_dir   = self.md_input_dir
			nnp_input_dir  = self.nnp_input_dir
			dft_input_dir  = self.dft_input_dir
			n2p2_path      = self.n2p2_path
			training_cores = self.training_cores
			training_queue = self.training_queue
			training_walltime=self.training_walltime
			n2p2_path      = self.n2p2_path
			lmp_path       = self.lmp_path
			atomsk_path    = self.atomsk_path
			md_cores       = self.md_cores
			md_queue       = self.md_queue         
			md_walltime    = self.md_walltime      
			dft_cores      = self.dft_cores
			dft_queue      = self.dft_queue        
			dft_walltime   = self.dft_walltime  
			dft_interval   = self.dft_interval

			scheduler      = self.scheduler
			modules        = self.modules

			with open(base_dir+'/log','w') as log:
				log.write('start the training iterations\n')

			for i in range(restart, iteration+1):

				if use_old_weight <= i:
					use_old = True
				else:
					use_old = False
				print("use_old", use_old_weight, i, use_old)

				base_name = str(i)				
				os.chdir(base_dir)
				new_db_dir = base_dir +'/1_data/data_'+base_name
				if not os.path.exists(new_db_dir):
					os.makedirs(new_db_dir)
					subprocess.call('cp -r '+db_dir+'/* '+new_db_dir,shell=True)
				subprocess.call('rm '+new_db_dir+'/input.data',shell=True)
				if ref_zbl == True:
					subprocess.call('cat '+new_db_dir+'/*_ref.data > '+new_db_dir+\
														 '/input.data',shell=True)
				elif ref_script != None:
					subprocess.call('cat '+new_db_dir+'/*_ref.data > '+new_db_dir+\
														 '/input.data',shell=True)
				else:
					subprocess.call('cat '+new_db_dir+'/*.data > '+new_db_dir+\
														 '/input.data',shell=True)

				traindir = base_dir + '/2_training/training_'+base_name
				try:
					os.stat(traindir)
				except:
					os.makedirs(traindir)
					os.chdir(traindir)
				qnums_n2p2  = multi_ANN_training(base_name=base_name,
												   base_dir=base_dir,
												   db_dir=new_db_dir,
												   nnp_input_dir=nnp_input_dir,
												   n2p2_path=n2p2_path,
												   modules=modules,
												   architect=NN_architect,
												   use_old=use_old,
												   ncores=training_cores,
												   queue=training_queue,
												   walltime=training_walltime,
												   scheduler=scheduler)
				
				#Tracking the multiple NNs
				submitted_n2p2 = sorted(qnums_n2p2.keys())
				flag_n2p2 = True
				while flag_n2p2:
					n2p2_job_done, error, job_failed = tracking_qjobs(qnums=qnums_n2p2,
					               types='n2p2_training',
								   base_dir=base_dir,
								   scheduler='slurm')
					finished_n2p2 = sorted(n2p2_job_done.keys())
					#print(submitted_n2p2)
					#print(finished_n2p2)
					if submitted_n2p2 == finished_n2p2:
						flag_n2p2 = False
						break

				training_dirs = list(qnums_n2p2.values())
				# Write a file indicating the job is done


				nnp_data_dirs = multi_collect_nnp_data(training_dirs)

				#Use one force field: Primary NN 
				Primary_nnp_data = nnp_data_dirs[0]
				print(nnp_data_dirs)
				print('Prime',Primary_nnp_data)
				qnums_lmp = lmp_ann_dynamics_cca(base_name=base_name,
				                             base_dir =base_dir,
											 nnp_data_dir=Primary_nnp_data,
											 md_input_dir=md_input_dir,
											 lmp_path = lmp_path,
											 ncores = md_cores,
											 queue = md_queue,
											 walltime = md_walltime,
											 scheduler = scheduler,
											 modules = modules)

				#Start the multiple-fit procedure
				
				#Run NNP single point
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

				#Submit DFT sp using vasp
				qnums_dft, dft_subdir, job_done = \
				dft_sp(base_name     = base_name,
					   base_dir      = base_dir,
					   job_done      = lmp_job_done_dftsp,
					   interval      = dft_interval,
					   dft_input_dir = dft_input_dir,
					   ncores        = dft_cores,
					   queue         = dft_queue,
					   walltime      = dft_walltime,
					   scheduler     = scheduler,
					   modules       = modules)

				submitted_dft = sorted(qnums_dft.keys())
				#print(submitted_dft)
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

				print(job_failed)
				if len(job_failed.values()) > 0:
					for key in job_failed.keys():
						v = job_failed[key]
						dft_job_done = {key:val for key, val in dft_job_done.items() if val != v}
						st = v[0].split("/")[:-1]
						v2 = "/".join(st)
						dft_subdir[v2].remove(int(key))

				if i in reduce_db.keys():
					eabove = reduce_db[i]
					db_reduce = True
				else:
				 	eabove = 0
				 	db_reduce = False

				#eabove = 0
				#db_reduce = False
				dft_job_done, dbnames = \
				format_dft_data(base_name=base_name,
						         base_dir=base_dir,
								 dft_subdir=dft_subdir,
								 dft_job_done=dft_job_done,
								 db_dir=db_dir,
								 eabove=eabove,
								 db_reduce=db_reduce)

				print(dbnames)

				if ref_zbl == True:
					data_files = []
					for dbname in dbnames:
						data_dbname = ref_zbl_DFT_data(dataname   = dbname, 
												   db_dir     = db_dir, 
												   lmp_path   = lmp_path, 
												   atomsk_path= atomsk_path)
						data_files.append(data_dbname)

				elif ref_script != None:
					data_files = []
					for dbname in dbnames:
						data_dbname = ref_script_DFT_data(dataname   = dbname, 
												   db_dir     = db_dir, 
												   script     = db_dir+'/'+ref_script,
												   lmp_path   = lmp_path, 
												   atomsk_path= atomsk_path)
						data_files.append(data_dbname)

				else:
					data_files = dbnames

				print(data_files)

				os.chdir(db_dir)


			return
			
	##################################################################################
			
	class Nitramine:
		def __init__(self, 
					 base_dir, db_dir, md_input_dir,
					 dft_input_dir, nnp_input_dir,
					 ffield, ffield_reax,
					 training_cores, training_queue, training_walltime,
					 md_cores,       md_queue,       md_walltime,
					 dft_cores,      dft_queue,      dft_walltime,
					 species_cores,  species_queue,  species_walltime,
					 commands, scheduler='slurm',
					 modules=['intel/17.0.1.132 impi/2017.1.132']):

			self.scheduler        = scheduler
			self.modules          = modules
			self.base_dir         = base_dir
			self.db_dir           = base_dir + "/" + db_dir
			self.md_input_dir     = base_dir + "/" + md_input_dir
			self.dft_input_dir    = base_dir + "/" + dft_input_dir
			self.nnp_input_dir    = base_dir + "/" + nnp_input_dir
			self.ffield           = base_dir + "/" + ffield
			self.ffield_reax      = base_dir + "/" + ffield_reax
			self.training_cores   = training_cores
			self.training_queue   = training_queue
			self.training_walltime= training_walltime
			self.md_cores         = md_cores
			self.md_queue         = md_queue
			self.md_walltime      = md_walltime
			self.dft_cores        = dft_cores
			self.dft_queue        = dft_queue
			self.dft_walltime     = dft_walltime
			self.species_cores    = species_cores
			self.species_queue    = species_queue
			self.species_walltime = species_walltime
			self.commands         = commands
			self.scheduler        = scheduler
			self.modules          = modules

			self.n2p2_path        = commands['n2p2']
			self.lmp_path         = commands['lmp']
			self.atomsk_path      = commands['atomsk']

		def Iteration(self, use_old_weight=5, 
				      reduce_db={}, 
					  iteration=20, 
					  restart=1, 
					  start_species=20):

			self.start_species    = start_species
			self.reduce_db        = reduce_db
			self.iteration        = iteration
			self.restart          = restart
			self.use_old_weight   = use_old_weight

			base_dir       = self.base_dir
			db_dir         = self.db_dir
			md_input_dir   = self.md_input_dir
			nnp_input_dir  = self.nnp_input_dir
			dft_input_dir  = self.dft_input_dir
			n2p2_path      = self.n2p2_path
			training_cores = self.training_cores
			training_queue = self.training_queue
			training_walltime=self.training_walltime
			n2p2_path      = self.n2p2_path
			lmp_path       = self.lmp_path
			atomsk_path    = self.atomsk_path
			md_cores       = self.md_cores
			md_queue       = self.md_queue         
			md_walltime    = self.md_walltime      
			dft_cores      = self.dft_cores
			dft_queue      = self.dft_queue        
			dft_walltime   = self.dft_walltime     
			species_cores  = self.species_cores    
			species_queue  = self.species_queue    
			species_walltime=self.species_walltime 
			scheduler      = self.scheduler
			modules        = self.modules
			ffield         = self.ffield
			ffield_reax    = self.ffield_reax

			with open(base_dir+'/log','w') as log:
				log.write('start the training iterations\n')

			for i in range(restart, iteration+1):
				base_name = str(i)

				if use_old_weight <= i:
					use_old = True
				else:
					use_old = False
				print("use_old", use_old_weight, i, use_old)
				os.chdir(base_dir)

				new_db_dir = base_dir+'/db_'+base_name
				if not os.path.exists(new_db_dir):
					subprocess.call('cp -r '+db_dir+' '+new_db_dir,shell=True)
				subprocess.call('rm '+new_db_dir+'/input.data',shell=True)
				subprocess.call('cat '+new_db_dir+'/*_ref.data > '+new_db_dir+\
													 '/input.data',shell=True)
				traindir = base_dir + '/training_'+base_name

				#Submit and Track NN training using n2p2
				training = n2p2_ANN_training_nitramine(base_name    =base_name,
												 base_dir     =base_dir,
												 db_dir       =new_db_dir,
												 nnp_input_dir=nnp_input_dir,
												 n2p2_path    =n2p2_path,
												 ncores       =training_cores,
												 queue        =training_queue,
												 walltime     =training_walltime,
												 use_old      =use_old,
												 scheduler    =scheduler,
												 modules      =modules)
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
				print(nnp_data_dir)

				#Submit and Track NN dyanmics using lammps+n2p2
				qnums_lmp = lmp_ann_dynamics(base_name    =base_name,
											 base_dir     =base_dir,
											 nnp_data_dir =nnp_data_dir,
											 md_input_dir =md_input_dir,
											 lmp_path     =lmp_path,
											 ncores       =md_cores, 
											 queue        =md_queue,
											 walltime     =md_walltime,
											 scheduler    =scheduler, 
											 modules      =modules)

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

				#Submit DFT sp using vasp
				qnums_dft, dft_subdir, job_done = \
				dft_sp_nitramine(base_name     = base_name,
					   base_dir      = base_dir,
					   job_done      = lmp_job_done_dftsp,
					   interval      = 2000,  # need to be controllable
					   dft_input_dir = dft_input_dir,
					   ncores        = dft_cores,
					   queue         = dft_queue,
					   walltime      = dft_walltime,
					   scheduler     = scheduler,
					   modules       = modules)
				
				#Species analysis before checking DFT sp calculations
				if int(base_name) >= self.start_species:
					exclude = Analyze_database(new_db_dir+'/input.data')
					qnums_species, species_subdir, job_done_species =\
					species_dft_dynamics(base_name     = base_name,
										 base_dir      = base_dir,
										 dft_input_dir = dft_input_dir,
										 lmp_job_done  = lmp_job_done_species,
										 ncores        = species_cores,
										 queue         = species_queue,
										 walltime      = species_walltime,
										 exclude       = exclude,
				 						 interval      = 2000,
										 ffield_reax   = ffield_reax,
										 ffoption      = '',
										 lmp_path      = lmp_path,
										 atomsk_path   = atomsk_path,
										 scheduler     = scheduler,
										 modules       = modules)

					with open(base_dir+'/log','a') as log:
						log.write("%s iteration species DFT dynamics\n" % base_name)
						keys = qnums_species.keys()
						for key in keys:
							log.write("%s %s\n" % (key, qnums_species[key]))
						log.write("___________________________________\n")

					submitted_dft_species = sorted(qnums_species.keys())

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


				if i in reduce_db.keys():
					eabove = reduce_db[i]
					db_reduce = True
				else:
				 	eabove = 0
				 	db_reduce = False

				dft_job_done, dbnames = \
				format_dft_data(base_name=base_name,
						         base_dir=base_dir,
								 dft_subdir=dft_subdir,
								 dft_job_done=dft_job_done,
								 db_dir=db_dir,
								 eabove=eabove,
								 db_reduce=db_reduce)

				if int(base_name) >= self.start_species:
					flag_species = True
					while flag_species:
						dft_job_done2, error, job_failed = tracking_qjobs(qnums    = qnums_species,
													   types    = 'vasp',
													   base_dir = base_dir,
													   scheduler= scheduler)
						finished_dft_species  = sorted(dft_job_done2.keys())

						print(len(submitted_dft_species), len(finished_dft_species))

						if submitted_dft_species == finished_dft_species:
							flag_dft_species = False
							break

					dft_job_done_, species_dbnames = \
					format_dft_data(base_name    = base_name,
							        base_dir     = base_dir,
									dft_subdir   = species_subdir,
									dft_job_done = dft_job_done2,
									db_dir       = db_dir,
									eabove       = eabove,
									db_reduce    = False)
	
				data_files = []
				for dbname in dbnames:
					data_dbname = ref_DFT_data(dataname   = dbname, 
							                   db_dir     = db_dir, 
											   ffield     = ffield, 
											   lmp_path   = lmp_path, 
											   atomsk_path= atomsk_path)
					data_files.append(data_dbname)
				if int(base_name) >= self.start_species:
					for dbname2 in species_dbnames:
						data_dbname = ref_DFT_data(dataname   = dbname, 
												   db_dir     = db_dir, 
												   ffield     = ffield, 
												   lmp_path   = lmp_path, 
												   atomsk_path= atomsk_path)
						data_files.append(data_dbname)
				os.chdir(db_dir)


			return

