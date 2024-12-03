#!/bin/env python
import subprocess 
import numpy as np
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

def data2images(datafile):
	images = []
	Nimages = 0
	data = open(datafile, 'r')
	data_contents = data.readlines()
	
	for i in range(len(data_contents)):
		line = data_contents[i]
		if line == "begin\n":
			Nimages += 1
			start = i
		
		elif line == "end\n":
			end = i
			singleimage = data_contents[start:end]
			cell_ = singleimage[2:5] 
			atom_lines = singleimage[5:-2]
			energy = float(singleimage[-2].split()[1]) * ( (kcal/mol) / eV)
			cell = []
			for l in cell_:
				cell.append([float(ll) for ll in l.split()[1:]])
			positions = []
			forces = []
			symbols = []
			for atom in atom_lines:
				a = atom.split()
				positions.append([float(p) for p in a[1:4]])
				symbols.append(a[4])
				forces.append([float(f) for f in a[7:]])
			forces = np.array(forces) * ( (kcal/mol) / eV)
			formula = "".join(symbols)
			atoms = Atoms(formula, cell=cell,
						  positions=positions,
						  pbc=[1,1,1])
			atoms.set_calculator(SPC(atoms, energy=energy, forces=forces))
			images.append(atoms)
	data.close()
	return images

datafile = sys.argv[1]
images = data2images(datafile)
#view(images)
dbfile = datafile[:-5] + ".db"
print(dbfile) 

