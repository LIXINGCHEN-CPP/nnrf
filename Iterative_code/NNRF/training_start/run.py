import NNRF
import os

#Define current directory
base_path = os.getcwd()

#Define path of n2p2 code, lammps, and atomsk
if os.path.exists('/scratch/brown'):
	n2p2_path = "/depot/lpl/apps/LAMMPS/7_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/n2p2_package"
	lmp_path = "/depot/lpl/apps/LAMMPS/7_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/src/lmp_mpi"
else:
	n2p2_path = "/depot/lpl/apps/LAMMPS/8_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132_rice/n2p2_package"
	lmp_path = "/depot/lpl/apps/LAMMPS/8_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132_rice/src/lmp_mpi"
	
atomsk_path="/depot/lpl/data/pilsun_files/SNAP/reax/atomsk/atomsk_b0.10.6_Linux-amd64/atomsk"

commands = {'n2p2':n2p2_path,
			'lmp':lmp_path,
			'atomsk':atomsk_path}

## Initialize the iteration
Model = NNRF.Iterative_Train.Nitramine(
			 base_dir = base_path,
			 db_dir='1_db',
			 md_input_dir='2_md_input',
			 dft_input_dir='3_dftsp_input',
			 nnp_input_dir='4_nnp_input',
			 ffield='1_db/ffield',
			 ffield_reax='1_db/ffield_2014',
			 training_cores=24,
			 training_queue='lpl',
			 training_walltime='336:00:00',
			 md_cores=24,
			 md_queue='lpl',
			 md_walltime='336:00:00',
			 dft_cores=24,
			 dft_queue='lpl',
			 dft_walltime='336:00:00',
			 species_cores=24,
			 species_queue='standby',
			 species_walltime='4:00:00',
			 commands=commands,
			 scheduler = 'slurm',
			 modules=['intel/17.0.1.132 impi/2017.1.132'])

## Start the training
Model.Iteration(use_old_weight = 5,
			    reduce_db={1:20,2:20,3:20},
			 	iteration=20,
			 	restart=1,
			 	start_species=10)

