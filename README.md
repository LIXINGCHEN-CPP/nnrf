# nnrf_nitramines: Training Data, NNRF Potentials, and Iterative Code for HE materials

## Neural network reactive force field for C, H, N, O systems  
 
### npj Comput Mater 7, 9 (2021).   
 Abstract:   
 Reactive force fields have enabled an atomic level description of a wide range of phenomena, from chemistry at extreme conditions to the operation of electrochemical devices and catalysis. While significant insight and semi-quantitative understanding have been drawn from such work, the accuracy of reactive force fields limits quantitative predictions. We developed a neural network reactive force field (NNRF) for CHNO systems to describe the decomposition and reaction of the high energy nitramine 1,3,5-Trinitroperhydro-1,3,5-triazine (RDX). NNRF was trained using energies and forces of a total of 3100 molecules (11941 geometries) and 15 condensed matter systems (32973 geometries) obtained from density functional theory calculations with semi-empirical corrections to dispersion interactions. The training set is generated via a semi-automated iterative procedure that enables refinement of the NNRF until a desired accuracy is attained. The RMS error of NNRF on a testing set of configurations describing the reaction of RDX is one order of magnitude lower than current state of the art potentials.    


## The NNRF Potentials
 The NNRF potentials for Gen1.x, Gen2.3 and Gen3.7 are provided and can be used for Molecule Dynamic simulations for Nitramines.    

### Gen1.X NNRF for thermal decomposition for RDX crystal
 Gen1.1 ~ Gen1.9 are for the thermal decomposition of RDX crytal. 

### Gen2.X NNRF for thermal decomposition for RDX crystal
 Gen2.8 is the NNRF for the thermal decomposition of RDX, HMX, NM, CL20, TNT, TATB, PETN crytals.  

### Gen3.X NNRF for Shock dynamics for PETN crystal and Liquid NM
 Gen3.7 is the NNRF for the shock dynamics of PETN crystal and Liquid NM.

## The Iterative code

 This is python module to run a iterative process for NNRF training.      

## The Training data

 This is the collection of data for every iteration of the training process for Gen1.X ~ Gen2.X
 Individual file is different geometry in the format of n2p2 data file.   
 Please check the input data format of n2p2 (https://compphysvienna.github.io/n2p2/Topics/cfg_file.html).   
 All individual files were concatenated to train the NNRF.
 The training data for shock response (Gen3.X) will be available soon.

### The energy and forces of data files are corresponding to E(DFT) - E(Reference Potential) and F(DFT) - F(Reference Potential).

 In the training of the NNRF, NNRF was designed to learn and predict this difference.   
 In the evaluation of the NNRF with trained parameters, NNRF + ReaxFF VC will be corresponding to PBE-D2 (Ground truth).  


### Installing the NNRF python module

 Add instruction



