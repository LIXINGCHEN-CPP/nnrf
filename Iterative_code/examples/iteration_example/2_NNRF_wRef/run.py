import NNRF
import os

#Define current directory
base_path = os.getcwd()

#Define path of n2p2 code, lammps, and atomsk
if os.path.exists('/scratch/brown'):
	n2p2_path = "/depot/lpl/apps/LAMMPS/7_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/n2p2_package"
	lmp_path = "/depot/lpl/apps/LAMMPS/7_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/src/lmp_mpi"
elif os.path.exists('/scratch/bell'):
	n2p2_path = "/depot/lpl/apps/LAMMPS/10_n2p2_lammps_bell_gcc_9.3.0_openmpi_3.1.4/n2p2_package"
	lmp_path = "/depot/lpl/apps/LAMMPS/10_n2p2_lammps_bell_gcc_9.3.0_openmpi_3.1.4/n2p2_package/src/interface/lammps-nnp/src/lmp_mpi"
else:
	n2p2_path = "/depot/lpl/apps/LAMMPS/6_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132_Halstead/n2p2_package"
	lmp_path = "/depot/lpl/apps/LAMMPS/6_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132_Halstead/src/lmp_mpi"

atomsk_path="/depot/lpl/data/pilsun_files/SNAP/reax/atomsk/atomsk_b0.10.6_Linux-amd64/atomsk"
python_path="/home/yoop/miniconda3/bin/python"

commands = {'n2p2':n2p2_path,
			'lmp':lmp_path,
			'atomsk':atomsk_path,
			'python':python_path}

## Initialize the iteration
Model = NNRF.Iterative_Train.NNRF_wRef(
			 base_dir   = base_path,
			 input_db   = base_path+'/input_db',
			 input_dft  = base_path+'/input_dft',
			 input_md   = base_path+'/input_md',
			 input_nnrf = base_path+'/input_nnp',
			 NN_architect = [[30,30]],
			 commands=commands,
			 scheduler = 'slurm')

for n in range(1,3):
	if n == 1:
		use_old_weight = False
	else:
		use_old_weight = True

	Model.Collect_DB(iteration=n, 
					  ref_pot=False)

	Model.N2P2_train(iteration=n, 
					  use_old_weight=use_old_weight,	
					  training_remote = {'cores':24,
										 'queue':'lpl',
										 'walltime':'24:00:00',
										 'modules':['intel/17.0.1.132', 'impi/2017.1.132'], #Brown/Halstead
										 #'modules':['gcc/9.3.0','openmpi/3.1.4','openblas/0.3.8','gsl/2.4'] #Bell
										 }
					 )

	Model.LMP_N2P2_dynamics(iteration=n,
							 md_remote = {'cores':24,
										  'queue':'lpl',
										  'walltime':'24:00:00',
										  'modules':['intel/17.0.1.132', 'impi/2017.1.132'], #Brown/Halstead
										  #'modules':['gcc/9.3.0','openmpi/3.1.4'],
										  }
							 )

	Model.vasp_DFT_MDtraj(iteration=n,
						   interval=1000,
						   dft_remote = {'cores':24,
										 'queue':'lpl',
										 'walltime':'24:00:00',
										 'modules':['intel/17.0.1.132', 'impi/2017.1.132'], #Brown/Halstead
										 #'modules':['intel/19.0.5.281','openmpi/3.1.4']
										 }
							)

	Model.Format_DFT_results(iteration=n,
							  reduce_db={})

	Model.DB_ref(iteration=n,ref_script=None)

