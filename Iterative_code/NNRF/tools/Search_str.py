import NNRF
from NNRF.lammps_sp import n2p2_single_point
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

images = Search_str(nnps=nnps, db='', commands=commands)



