#!/bin/env python
from NNRF.species import Dynamics_species, Search_Bonding_species, sort_species, bond_analysis
from ase.io import read, write
import os

#Define path of n2p2 code, lammps, and atomsk
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

output = Dynamics_species(trajectory='Gen1-8_8RDX_3_2500K_200ps_dynamics_200000_2x2x2.xyz', 
		         working_dir='./species_2x2x2', 
				 interval=100, 
				 ffield_reax='ffield_2014',
				 ffoption= '',
				 exclude = [],
				 lmp_path=lmp_path,
				 atomsk_path=atomsk_path,
				 rdict = {},
				 ReaxFFBO=True)

"""
images, botables = sort_species(xyzfile='total.xyz', bofile='total_bo.reaxc')
write('./species_sorted.xyz',images)
with open('./species_sorted_bo.reaxc','w') as sbo:
	bo_str = "".join(botables)
	sbo.write(bo_str)

bond_infos = bond_analysis(images, botables)
write('./species_reduced.xyz',bond_infos[:,0])


"""
