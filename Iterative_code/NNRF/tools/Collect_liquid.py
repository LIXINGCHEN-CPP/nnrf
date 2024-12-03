import NNRF
from NNRF.lammps_sp import n2p2_single_point, Search_str
from ase.io import read, write
import numpy as np
import os, random, copy

base_path = os.getcwd()

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

nnps = ['./nnp-data_1',
	    './nnp-data_2',
		'./nnp-data_3']

images = Search_str(nnps=nnps, db_file='Liquid_4000K_NbMoTaW_100.db', commands=commands, cores=20)

"""
atoms = read('POSCAR_1')
chem = atoms.get_chemical_symbols()

nnp_ermse, nnp_frmse = n2p2_single_point(nnps, atoms, commands)
ermse1 = np.sqrt( ( nnp_ermse[1] - nnp_ermse[0] ) ** 2 ) / len(atoms)
ermse2 = np.sqrt( ( nnp_ermse[2] - nnp_ermse[0] ) ** 2 ) / len(atoms)
frmse1 = np.sqrt( np.mean( ( nnp_frmse[1] - nnp_frmse[0] ) ** 2 ) )
frmse2 = np.sqrt( np.mean( ( nnp_frmse[2] - nnp_frmse[0] ) ** 2 ) )

c_1 = abs(ermse1-ermse2)
c_2 = abs(frmse1-frmse2)
print(c_1, c_2)
	
images = []
track = []

while len(images) < 11:
	chem_2 = copy.deepcopy(chem)
	atoms_2 = copy.deepcopy(atoms)
	random.shuffle(chem_2)
	atoms_2.set_chemical_symbols(chem_2)
	atoms_2.rattle(stdev=random.uniform(0.05, 0.20), seed=random.randint(0,10000))

	syms = "".join(chem_2)

	nnp_ermse, nnp_frmse = n2p2_single_point(nnps, atoms_2, commands)
	ermse1 = np.sqrt( ( nnp_ermse[1] - nnp_ermse[0] ) ** 2 ) / len(atoms_2)
	ermse2 = np.sqrt( ( nnp_ermse[2] - nnp_ermse[0] ) ** 2 ) / len(atoms_2)
	frmse1 = np.sqrt( np.mean( ( nnp_frmse[1] - nnp_frmse[0] ) ** 2 ) )
	frmse2 = np.sqrt( np.mean( ( nnp_frmse[2] - nnp_frmse[0] ) ** 2 ) )
	print(abs( ermse1 - ermse2 ), abs( frmse1 - frmse2 ))
	if (abs( ermse1 - ermse2 ) > c_1) or (abs( frmse1 - frmse2 ) > c_2):
		images.append(atoms_2)

	print(len(images))

write('Collect_3.xyz',images, format='xyz')


##
Model = NNRF.Iterative_Train.ComplexCompounAlloy(
		base_dir=base_path,
		db_dir='1_db',
		md_input_dir='2_md_input',
		dft_input_dir='3_dftsp_input',
		nnp_input_dir='4_nnp_input',
		training_cores=20, 
		training_queue='lpl', 
		training_walltime="336:00:00",
		md_cores=20,       
		md_queue='lpl',       
		md_walltime="336:00:00",
		dft_cores=20,      
		dft_queue='lpl',      
		dft_walltime="336:00:00",
		commands=commands, 
		scheduler='slurm',
		modules=['intel/17.0.1.132 impi/2017.1.132'])

Model.Iteration(NN_architect=[[20,20],[10,10,10]],
	            iteration=20,
				restart=1,
				use_old_weight=10,
				mf_process=20)
"""

