#Molecular recognition through binning and relative positions
#Takes dump file-frame as command line input, use with parent.sh and child.sh 
#Do not cross 10 atoms a bin, automated bin sizing based on cut offs should ensure

#!/usr/bin/env python
import os
import sys
import math
import numpy
from numpy import zeros
import random
import re

def atoi(text):
	if(text.isdigit() == True):
		return int(text)
	else:
		return text

def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)',text) ]

def dists(x2,x1,y2,y1,z2,z1): #Returns least distance between atoms or periodic images
	d1=((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)**0.5
	d2=((x2+xrng-x1)**2+(y2-y1)**2+(z2-z1)**2)**0.5
	d3=((x2-xrng-x1)**2+(y2-y1)**2+(z2-z1)**2)**0.5
	d4=((x2-x1)**2+(y2+yrng-y1)**2+(z2-z1)**2)**0.5
	d5=((x2-x1)**2+(y2-yrng-y1)**2+(z2-z1)**2)**0.5
	d6=((x2-x1)**2+(y2-y1)**2+(z2+zrng-z1)**2)**0.5
	d7=((x2-x1)**2+(y2-y1)**2+(z2-zrng-z1)**2)**0.5
	d8=((x2+xrng-x1)**2+(y2+yrng-y1)**2+(z2-z1)**2)**0.5
	d9=((x2-xrng-x1)**2+(y2-yrng-y1)**2+(z2-z1)**2)**0.5
	d10=((x2-x1)**2+(y2+yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d11=((x2-x1)**2+(y2-yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	d12=((x2+xrng-x1)**2+(y2-y1)**2+(z2+zrng-z1)**2)**0.5
	d13=((x2-xrng-x1)**2+(y2-y1)**2+(z2-zrng-z1)**2)**0.5
	d14=((x2+xrng-x1)**2+(y2+yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d15=((x2-xrng-x1)**2+(y2-yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	d16=((x2+xrng-x1)**2+(y2-yrng-y1)**2+(z2-z1)**2)**0.5
	d17=((x2-xrng-x1)**2+(y2+yrng-y1)**2+(z2-z1)**2)**0.5
	d18=((x2-x1)**2+(y2+yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	d19=((x2-x1)**2+(y2-yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d20=((x2+xrng-x1)**2+(y2-y1)**2+(z2-zrng-z1)**2)**0.5
	d21=((x2-xrng-x1)**2+(y2-y1)**2+(z2+zrng-z1)**2)**0.5
	d22=((x2+xrng-x1)**2+(y2+yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	d23=((x2+xrng-x1)**2+(y2-yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d24=((x2-xrng-x1)**2+(y2+yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d25=((x2-xrng-x1)**2+(y2-yrng-y1)**2+(z2+zrng-z1)**2)**0.5
	d26=((x2+xrng-x1)**2+(y2-yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	d27=((x2-xrng-x1)**2+(y2+yrng-y1)**2+(z2-zrng-z1)**2)**0.5
	return min(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27)

### INPUT PARAMETERS TO CHANGE ###
natoms=464 # Number of atoms in system
dumpfreq=10000 # Dump frequency

Dump_files = []
for root, dirs, files in os.walk("Dumpfiles/."):
	for filename in files:
		if('dump' in filename): # Looks for keyword shared by all dump frames
			Dump_files.append(filename)
Dump_files.sort(key=natural_keys)
print('%s number of frames to be analyzed' %len(Dump_files))

ofile=open("dump_from_bond_table.reaxc", 'w')
for k in range(len(Dump_files)):
	t = k*dumpfreq
	print("Starting timestep %s, %.1f percent complete" %(t, round(100.*k/len(Dump_files),1)))
	path = "Dumpfiles/"+str(Dump_files[k])
	ifile = open(path, 'r')
	# Assume CHNO matching dump frame and bond table
	#rdict = {'11': '1.51', '12': '2.00','21': '2.00', '13': '1.65','31': '1.65','14': '1.65', '41': '1.65','15': '1.25', '51': '1.25','22': '1.7','23': '1.7','32': '1.7', '24': '1.4','42': '1.4', '25': '1.74','52': '1.74','33': '1.7', '34': '1.37','43': '1.37', '35': '1.74','53': '1.74','44': '1.6', '45': '1.65','54': '1.65', '55': '1.75'}
	rdict = {'22': '1.51', '21': '2.00','12': '2.00', '23': '1.65','32': '1.65','24': '1.65', '42': '1.65','25': '1.25', '52': '1.25','11': '1.7','13': '1.7','31': '1.7', '14': '1.4','41': '1.4', '15': '1.74','51': '1.74','33': '1.7', '34': '1.37','43': '1.37', '35': '1.74','53': '1.74','44': '1.6', '45': '1.65','54': '1.65', '55': '1.75'}
	fudge_fact=1.0
	
	x = zeros(natoms, float)
	y = zeros(natoms, float)
	z = zeros(natoms, float)
	typ = zeros(natoms, int)
	ids=zeros(natoms, int)
	j=0
	i=0
	for line in ifile:
		if(j>=9):   
			items = str.split(line)
			ids[i]=eval(items[0])
			typ[i] = eval(items[2])
			x[i] = eval(items[3])
			y[i] = eval(items[4])
			z[i] = eval(items[5])
			i=i+1
		j=j+1
	ifile.close()
	xmin=min(x)
	xmax=max(x)
	xrng=xmax-xmin
	ymin=min(y)
	ymax = max(y)
	yrng=ymax-ymin
	zmin = min(z)
	zmax = max(z)				
	zrng=zmax-zmin
	
	bw=1.85*fudge_fact+0.1 #Use smallest possible bin size
	nbinats=15
	ax=bw #in A
	ay=bw #in A
	az=bw #in A
	nx=int((xmax - xmin)/ax)# + 1
	ny=int((ymax - ymin)/ay)# + 1
	nz=int((zmax - zmin)/az)# + 1
	totats=zeros([nx,ny,nz], int)
	atinfo=zeros([nx,ny,nz,nbinats,5], float)
	
	for i in range (natoms):
		b1=int((x[i]-xmin)/ax)
		b2=int((y[i]-ymin)/ay)
		b3=int((z[i]-zmin)/az)
		if(b1>=nx):b1=nx-1
		if(b2>=ny):b2=ny-1
		if(b3>=nz):b3=nz-1
		atinfo[b1][b2][b3][totats[b1][b2][b3]][0]=ids[i]
		atinfo[b1][b2][b3][totats[b1][b2][b3]][1]=typ[i]
		atinfo[b1][b2][b3][totats[b1][b2][b3]][2]=x[i]
		atinfo[b1][b2][b3][totats[b1][b2][b3]][3]=y[i]
		atinfo[b1][b2][b3][totats[b1][b2][b3]][4]=z[i]
		totats[b1][b2][b3]+=1
	  
	nbin=nx*ny*nz+1				#nominal value to initialise arrays
	
	ofile.write("# Timestep %s \n" %(t))
	ofile.write("# \n")
	ofile.write("# Number of particles %s \n" %(natoms))
	ofile.write("# \n")
	ofile.write("# Max number of bonds per atom 4 with coarse bond order cutoff 0.300 \n")
	ofile.write("# Particle connection table and bond orders \n")
	ofile.write("# id type nb id_1...id_nb \n")
	for b1 in range(nx):

		print(b1," of ", nx, '\n')
		for b2 in range(ny):
			for b3 in range(nz):
				for l in range(totats[b1][b2][b3]):
					alist=[]
					alist.append(int(atinfo[b1][b2][b3][l][0]))#Append id and type of atom in question to the list
					alist.append(int(atinfo[b1][b2][b3][l][1]))
					for xi in range(-1,2):
						for yi in range(-1,2):
							for zi in range(-1,2):
								B1=b1+xi
								B2=b2+yi
								B3=b3+zi
								if(B1>=nx):B1=B1%nx
								if(B2>=ny):B2=B2%ny
								if(B3>=nz):B3=B3%nz
					   
								for k in range(totats[B1][B2][B3]):
									dist=dists(atinfo[B1][B2][B3][k][2],atinfo[b1][b2][b3][l][2],atinfo[B1][B2][B3][k][3],atinfo[b1][b2][b3][l][3],atinfo[B1][B2][B3][k][4],atinfo[b1][b2][b3][l][4])
									lookup=str(int(atinfo[b1][b2][b3][l][1]))+str(int(atinfo[B1][B2][B3][k][1]))
									if(dist<=float(rdict[lookup])*fudge_fact and int(atinfo[b1][b2][b3][l][0])!=int(atinfo[B1][B2][B3][k][0])): alist.append(int(atinfo[B1][B2][B3][k][0]))
					  
					alist.insert(2, len(alist)-2) #Insert number of bonded atoms in col 3
					outstr=' '.join(map(str, alist))+"\n" #Write list of connected atoms
					ofile.write(outstr)
	ofile.write("# \n")
ofile.write("# ")
ofile.close()
