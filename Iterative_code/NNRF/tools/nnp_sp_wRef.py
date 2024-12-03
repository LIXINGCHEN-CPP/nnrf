import NNRF
from NNRF.lammps_sp import nnp_parity_wRef
from ase.io import read, write
import numpy as np
import os, random, copy, glob, subprocess
base_path = os.getcwd()
if os.path.exists('/scratch/brown'):
	n2p2_path = "/depot/lpl/apps/LAMMPS/7_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/n2p2_package"
	lmp_path = "/depot/lpl/apps/LAMMPS/7_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132/src/lmp_mpi"
elif os.path.exists('/scratch/bell'):
	n2p2_path = "/depot/lpl/apps/LAMMPS/10_n2p2_lammps_bell_gcc_9.3.0_openmpi_3.1.4/n2p2_package"
	lmp_path = "/depot/lpl/apps/LAMMPS/10_n2p2_lammps_bell_gcc_9.3.0_openmpi_3.1.4/n2p2_package/src/interface/lammps-nnp/src/lmp_mpi"
else:
	n2p2_path = "/depot/lpl/apps/LAMMPS/8_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132_rice/n2p2_package"
	lmp_path = "/depot/lpl/apps/LAMMPS/8_n2p2_lammps-16Mar18-intel-17.0.1.132-impi-2017.1.132_rice/src/lmp_mpi"
atomsk_path="/depot/lpl/data/pilsun_files/SNAP/reax/atomsk/atomsk_b0.10.6_Linux-amd64/atomsk"
python_path="/home/yoop/miniconda3/bin/python"

commands = {'n2p2':n2p2_path,
			'lmp':lmp_path,
			'atomsk':atomsk_path,
			'python':python_path}

nnps = ['./NNRF_parameters/nnp-data_Gen1-10']


nnp_parity_wRef(nnps=nnps, 
				  db_file='EchemDID_small_RS100_300K_v10.db', 
				  commands=commands, 
				  cores=1,interval=1,
				  input_file='ref_script.in',
				  elements=['O','Si','Cu'])

	


