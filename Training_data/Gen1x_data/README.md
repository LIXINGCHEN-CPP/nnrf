# NNRF Gen2.3 training data

### The training data files for NNRF Gen2.3 (include all data from Gen1.1 ~ Gen1.9 to Gen2.3).
 Individual file is different geometry in the format of n2p2 data file.  
 Please check the input data format of n2p2 (https://compphysvienna.github.io/n2p2/Topics/cfg_file.html). 

### The energy and forces of data files are corresponding to E(PBE-D2) - E(ReaxFF VC) and F(PBE-D2) - F(ReaxFF VC).
 In the training of the NNRF, NNRF was designed to learn and predict this difference.  
 In the evaluation of the NNRF with trained parameters, NNRF + ReaxFF VC will be corresponding to PBE-D2 (Ground truth).  
