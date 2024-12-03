from ase.units import eV, kcal, mol
from ase import Atoms, Atom
from itertools import combinations
from ase.io import read,write
from ase.data import atomic_numbers, atomic_masses
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator as SPC
import os,subprocess,glob,collections
import numpy as np
from .species import Rcutoff_botable, sort_species, bond_analysis, Species_Rec
from .lammpslib import write_lammps_data
from .convert import convert_dump_to_xyz_losing, convert_dump_to_xyz_complete


class Iterative_Train_jupyter:
    
    class nanoHUB:
        def __init__(self,base_dir, input_db, input_dft, input_md, input_nnrf, reaxff, reaxff_ref):
        
            self.base_dir = base_dir
            self.input_db = input_db
            self.input_md = input_md
            self.input_dft = input_dft
            self.input_nnrf = input_nnrf
            self.reaxff = base_dir+'/'+reaxff
            self.reaxff_ref = base_dir+'/'+reaxff_ref

        def Collect_DB(self,iteration,ref_pot):
            base_dir = self.base_dir
            input_db = self.input_db
            os.chdir(base_dir)
            db_dir = base_dir + '/1_data'
            Iter_db_dir = db_dir + '/data_%d' % (iteration)
            self.db_dir = db_dir
            self.Iter_db_dir = Iter_db_dir
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)
            if not os.path.exists(Iter_db_dir):
                subprocess.call('cp -r '+input_db+' '+Iter_db_dir, shell=True)
            os.chdir(Iter_db_dir)
            if os.path.exists(Iter_db_dir+'/input.data'):
                subprocess.call('rm input.data',shell=True)
            if ref_pot:
                subprocess.call('cat *_ref.data > input.data',shell=True)
            else:
                subprocess.call('cat *.data > input.data',shell=True)

            return Iter_db_dir

        def N2P2_train(self,iteration, use_old_flag, training_queue):
            base_dir = self.base_dir
            Iter_db_dir = self.Iter_db_dir
            input_nnrf = self.input_nnrf
            os.chdir(base_dir)

            #Copy 'input.nn' and 'input.data'
            training_dir = base_dir + '/2_training'
            self.training_dir = training_dir
            Iter_training_dir = training_dir + '/training_%d' % (iteration)
            self.Iter_training_dir = Iter_training_dir
            if not os.path.exists(training_dir):
                os.makedirs(training_dir)
            if not os.path.exists(Iter_training_dir):
                os.makedirs(Iter_training_dir)
                subprocess.call('cp '+Iter_db_dir+'/input.data '+Iter_training_dir, shell=True)
                subprocess.call('cp '+input_nnrf+'/input.nn '+Iter_training_dir, shell=True)
            os.chdir(Iter_training_dir)

            #Change parameter to use_old_weight in input.nn file
            use_old_weight(Iter_training_dir, use_old_flag=use_old_flag)

            #Start training using nnp-norm, nnp-scaling, nnp-train at the remote queue
            Launch_training(Iter_training_dir, training_queue)

            #Collect the parameters
            Iter_nnp_data_dir = collect_nnp_data(Iter_training_dir)
            self.Iter_nnp_data_dir = Iter_nnp_data_dir

            return Iter_nnp_data_dir

        def LMP_N2P2_dynamics(self,iteration, md_queue):
            base_dir = self.base_dir
            input_md = base_dir + '/' +self.input_md
            Iter_nnp_data_dir = self.Iter_nnp_data_dir
            
            os.chdir(base_dir)
            
            #Make MD directory and copy directories in input_md
            dynamics_dir = base_dir + '/3_NNmd'
            self.dynamics_dir = dynamics_dir
            Iter_dynamics_dir = dynamics_dir +'/md_test_%d' % (iteration)
            self.Iter_dynamics_dir = Iter_dynamics_dir

            if not os.path.exists(dynamics_dir):
                os.makedirs(dynamics_dir)
            if not os.path.exists(Iter_dynamics_dir):
                os.makedirs(Iter_dynamics_dir)
            
                #Copy the trainining parameters to individual md directory
                subprocess.call('cp -r '+input_md+'/* '+Iter_dynamics_dir,shell=True)
                md_dirs = glob.glob(Iter_dynamics_dir+'/*')
                print(md_dirs)
                md_count = 1
                for md_dir in md_dirs:
                    os.chdir(md_dir)
                    if not os.path.exists(md_dir+'/Done'):
                        subprocess.call('cp -r '+Iter_nnp_data_dir+' '+md_dir,shell=True)
                        #Start MD dyanmics using lammps at the remote queue
                        print("Launch MD simulation (%d/%d)" % (md_count, len(md_dirs)))
                        Launch_dynamics(md_dir, md_queue)
                    os.chdir(base_dir)
                    md_count += 1

            md_dirs = glob.glob(Iter_dynamics_dir+'/*')
            md_count = 1
            for md_dir in md_dirs:
                os.chdir(md_dir)
                outputs= sorted(glob.glob('*.stdout'), key=lambda x:int(x[:-7]))
                errors = sorted(glob.glob('*.stderr'), key=lambda x:int(x[:-7]))
                if len(outputs) >= 1:
                    output = outputs[-1]
                    success = check_lmp_out(output)
                    dump_file = 'dynamics.dump'
                    dump_xyz  = 'dynamics.xyz'
                    if success and os.path.exists(dump_file):
                        subprocess.call('touch Done',shell=True)
                        print("Convert dump file to xyz file (%d/%d)" % (md_count, len(md_dirs)))
                        #Collect the results and convert dynamics.dump to dynamics_*.xyz
                        Nimages = convert_dump_to_xyz_complete(dump_file,dump_xyz)
                        new_dump_xyz = dump_xyz[:-4] + '_%d.xyz' % Nimages
                        if not os.path.exists(new_dump_xyz):
                            subprocess.call('cp '+dump_xyz+' '+new_dump_xyz,shell=True)
                        subprocess.call('rm '+dump_xyz,shell=True)
                        md_count += 1

            return Iter_dynamics_dir
                

        def QE_DFT_MDtraj(self, iteration, interval, dft_queue):
            base_dir = self.base_dir
            input_dft= base_dir + '/'+ self.input_dft
            Iter_dynamics_dir = self.Iter_dynamics_dir
            #Make DFT directory
            DFT_dir = base_dir + '/4_DFT_MDtraj'
            Iter_DFT_dir = DFT_dir +'/DFTsp_%d' % iteration
            md_dir_names = glob.glob(Iter_dynamics_dir+'/*')
            dir_names = [md_dir.split('/')[-1] for md_dir in md_dir_names]
            if not os.path.exists(DFT_dir):
                os.makedirs(DFT_dir)
            if not os.path.exists(Iter_DFT_dir):
                os.makedirs(Iter_DFT_dir)
                for i in range(len(dir_names)):
                    dir_name = dir_names[i]
                    md_dir_name = md_dir_names[i]
                    os.makedirs(Iter_DFT_dir+'/'+dir_name)
                    dft_subdir = Iter_DFT_dir+'/'+dir_name
                    subprocess.call('cp '+md_dir_name+'/dynamics_*.xyz '+dft_subdir,shell=True)
                    subprocess.call('cp '+input_dft+'/qe_dftsp.in '+dft_subdir,shell=True)
                    subprocess.call('cp '+input_dft+'/*.UPF '+dft_subdir,shell=True)
            
            for i in range(len(dir_names)):
                dir_name = dir_names[i]
                dft_subdir = Iter_DFT_dir+'/'+dir_name
                os.chdir(dft_subdir)
                dyn = glob.glob('dynamics_*.xyz')[0]
                xyz_images = read(dyn,index=':')
                dft_count = 1
                total = len(xyz_images)/interval + 1
                for j in range(0,len(xyz_images),interval):
                    os.chdir(dft_subdir)
                    atoms = xyz_images[i]
                    dft_spdir = dft_subdir+'/%d_%d' % (j, interval)
                    if not os.path.exists(dft_spdir):
                        os.makedirs(dft_spdir)
                        subprocess.call('cp qe_dftsp.in '+dft_spdir,shell=True)
                        subprocess.call('cp *.UPF '+dft_spdir,shell=True)
                        os.chdir(dft_spdir)
                        atoms.write('structure.xyz')
                        write_qe_input(atoms,'qe_dftsp.in')

                    os.chdir(dft_spdir)
                    #Launch QE calculation
                    if not os.path.exists('Done'):
                        print("submit QE calculation of (%d/%d) %s with DFT (%d/%d)" % (i+1,len(dir_names),dir_name,dft_count,total))
                        Launch_QE(dft_spdir, 'qe_dftsp.in', dft_queue)
                        #Check if calculation is successfully done
                        outputs= sorted(glob.glob('*.stdout'), key=lambda x:int(x[:-7]))
                        errors = sorted(glob.glob('*.stderr'), key=lambda x:int(x[:-7]))
                        if len(outputs) >= 1:
                            output = outputs[-1]
                            success = check_qe_out(output)
                            #write Done file
                            if success:
                                subprocess.call('touch Done',shell=True)
                                subprocess.call('rm -r out',shell=True)
                    print("QE calculation of (%d/%d) %s with DFT (%d/%d) is completed" % (i+1,len(dir_names),dir_name,dft_count,total))
                    dft_count += 1

            #retrieve energy and forces
            self.Iter_DFT_dir = Iter_DFT_dir

            return Iter_DFT_dir

        def QE_DFT_Species(self, iteration, interval, lmp_path, species_queue):
            base_dir = self.base_dir
            input_dft= base_dir + '/'+ self.input_dft
            reaxff = self.reaxff
            ReaxFFBO = True
            ffoption = ''
            rdict = {}
            Iter_dynamics_dir = self.Iter_dynamics_dir

            #Make DFT directory
            Species_dir = base_dir + '/5_DFT_Species'
            Iter_Species_dir = Species_dir +'/Species_%d' % iteration

            total_xyz_files = []
            total_bo_files = []
            species_xyz = []
            jobname = 'DFT_species_dyn'

            md_dir_names = glob.glob(Iter_dynamics_dir+'/*')
            dir_names = [md_dir.split('/')[-1] for md_dir in md_dir_names]
            if not os.path.exists(Species_dir):
                os.makedirs(Species_dir)
            if not os.path.exists(Iter_Species_dir):
                os.makedirs(Iter_Species_dir)

            for i in range(len(dir_names)):
                dir_name = dir_names[i]
                md_dir_name = md_dir_names[i]
                species_subdir = Iter_Species_dir+'/'+dir_name
                if not os.path.exists(species_subdir):
                    os.makedirs(species_subdir)
                    subprocess.call('cp '+md_dir_name+'/dynamics_*.xyz '+species_subdir,shell=True)
                    subprocess.call('cp '+reaxff+' '+species_subdir,shell=True)
                    reaxff2 = species_subdir+'/'+reaxff.split('/')[-1]
                    os.chdir(species_subdir)
                    dynfile = glob.glob('./dynamics_*.xyz')
                    dyn_trajector = dynfile[0]
                    Dynamics_species_nanoHUB(trajectory = dyn_trajector,
                                     working_dir=species_subdir,
                                     ffield_reax=reaxff2,
                                     ffoption=ffoption,
                                     interval=interval,
                                     exclude = [],
                                     lmp_path=lmp_path,
                                     rdict=rdict,
                                     ReaxFFBO=ReaxFFBO)
                total_xyz_files.append(species_subdir+'/total.xyz')
                total_bo_files.append(species_subdir+'/total_bo.reaxc')

            species_dir = Iter_Species_dir+'/species'
            species_images = species_dir+'/species_reduced.xyz'

            if not os.path.exists(species_dir):
                os.makedirs(species_dir)

            os.chdir(species_dir)
            if not os.path.exists(species_images):
                convert1 = "cat "+" ".join(total_xyz_files)+" > "+species_dir+"/species.xyz"
                convert2 = "cat "+" ".join(total_bo_files) +" > "+species_dir+"/species_bo.reaxc"
                subprocess.call(convert1,shell=True)
                subprocess.call(convert2,shell=True)

                if ReaxFFBO:
                    images, botables = sort_species(xyzfile=species_dir+'/species.xyz', bofile=species_dir+'/species_bo.reaxc')
                    write(species_dir+'/species_sorted.xyz',images)
                    with open(species_dir+'/species_sorted_bo.reaxc','w') as sbo:
                        bo_str = "".join(botables)
                        sbo.write(bo_str)
                    bond_infos = bond_analysis(images, botables)
                    write(species_dir+'/species_reduced.xyz',bond_infos[:,0])
                else:
                    images = read('species.xyz',index=':')
                    images_sorted = sorted(images, key=lambda x:len(x))
                    images_reduced = [atoms for atoms in images_sorted if len(atoms) > 1 and len(atoms) < 100]
                    write(species_dir+'/species_reduced.xyz',images_reduced)

            species_structure = read(species_images,index=":")
            species_count = 1
            for i in range(len(species_structure)):
                dft_md_dir = species_dir+'/%d' % (i+1)
                species_atoms = species_structure[i]
                species_atoms.set_pbc([1,1,1])
                if not os.path.exists(dft_md_dir):
                    os.makedirs(dft_md_dir)

                    subprocess.call('cp '+input_dft+'/qe_dftmd_species.in '+dft_md_dir,shell=True)
                    subprocess.call('cp '+input_dft+'/*.UPF '+dft_md_dir,shell=True)
                    species_atoms.write(dft_md_dir+'/in_structure.xyz')
                    os.chdir(dft_md_dir)
                    write_qe_input(species_atoms,'qe_dftmd_species.in')

                os.chdir(dft_md_dir)
                if not os.path.exists('Done'):
                    print("submit QE-MD calculation for species of DFT (%d/%d)" % (i+1,len(species_structure)))
                    Launch_QE(dft_md_dir, 'qe_dftmd_species.in', species_queue)
                    #Check if calculation is successfully done
                    outputs= sorted(glob.glob('*.stdout'), key=lambda x:int(x[:-7]))
                    errors = sorted(glob.glob('*.stderr'), key=lambda x:int(x[:-7]))
                    if len(outputs) >= 1:
                        output = outputs[-1]
                        success = check_qe_out(output)
                        #write Done file
                        if success:
                            subprocess.call('touch Done',shell=True)
                            #subprocess.call('rm -r out',shell=True)
                print("Completed QE-MD calculations for species of DFT (%d/%d)" % (i+1,len(species_structure)))
                species_count += 1

            #retrieve energy and forces
            self.Iter_Species_dir = Iter_Species_dir

            return Iter_Species_dir
            

        def Collect_DFT_results(self, iteration):
            base_dir = self.base_dir
            Iter_DFT_dir = self.Iter_DFT_dir
            Iter_Species_dir = self.Iter_Species_dir
            iteration+=1
            #make new db dir
            db_dir = self.db_dir
            Iter_db_dir = self.Iter_db_dir
            new_Iter_db_dir = db_dir +'/data_%d' % iteration
            if os.path.exists(new_Iter_db_dir):
                collect_DB = new_Iter_db_dir
            else:
                subprocess.call('cp -r '+Iter_db_dir+' '+new_Iter_db_dir,shell=True)
                subprocess.call('rm '+new_Iter_db_dir+'/input.data',shell=True)
                collect_DB = new_Iter_db_dir
                
            collect_DB_names = []
            if Iter_DFT_dir != None:
                subdirs = glob.glob(Iter_DFT_dir+'/*')
                subdir_names = [name.split("/")[-1] for name in subdirs]
                for i in range(len(subdirs)):
                    os.chdir(subdirs[i])
                    db_name = collect_DB+'/'+subdir_names[i]+'_%d.db' % (iteration-1)
                    if not os.path.exists(db_name):
                        db = connect(db_name)
                        dftsp_dirs = glob.glob(subdirs[i]+"/*")
                        dftsp_dirs_ = [d for d in dftsp_dirs if os.path.isdir(d)]
                        dftsp_dirs_ = sorted(dftsp_dirs_,key=lambda x: x.split('/')[-1])
                        for j in range(len(dftsp_dirs_)):
                            dft_spdir = dftsp_dirs_[j]
                            os.chdir(dft_spdir)
                            #Check if calculation is successfully done
                            outputs= sorted(glob.glob('*.stdout'), key=lambda x:int(x[:-7]))
                            errors = sorted(glob.glob('*.stderr'), key=lambda x:int(x[:-7]))
                            success = False
                            output = None
                            if len(outputs) >= 1:
                                output = outputs[-1]
                                success = check_qe_out(output)
                            if success:
                                atoms = QE_sp_result(dft_spdir,output)
                                db.write(atoms)
                            else:
                                print('check:',dft_spdir)
                    collect_DB_names.append(db_name)
            if Iter_Species_dir != None:
                db_name = collect_DB+'/'+'species_%d.db' % (iteration-1)
                db = connect(db_name)
                speciesmd_dirs  = glob.glob(Iter_Species_dir+'/species/*')
                speciesmd_dirs_ = [s for s in speciesmd_dirs if os.path.isdir(s)]
                speciesmd_dirs_ = sorted(speciesmd_dirs_,key=lambda x: x.split('/')[-1])
                for j in range(len(speciesmd_dirs_)):
                    speciesmd_dir = speciesmd_dirs_[j]
                    os.chdir(speciesmd_dir)
                    #Check if calculation is successfully done
                    outputs= sorted(glob.glob('*.stdout'), key=lambda x:int(x[:-7]))
                    errors = sorted(glob.glob('*.stderr'), key=lambda x:int(x[:-7]))
                    success = False
                    output = None
                    if len(outputs) >= 1:
                        output = outputs[-1]
                        success = check_qe_out(output)
                    if success:
                        #Collect last structure
                        atoms = QE_md_result(speciesmd_dir,output)
                        db.write(atoms)
                    else:
                        print('check:',speciesmd_dir)
                collect_DB_names.append(db_name)

            self.collect_DB_names = collect_DB_names
            return collect_DB_names

     
        def DB_ref(self, iteration, lmp_path):
            base_dir = self.base_dir
            db_dir = self.db_dir
            reaxff_ref = self.reaxff_ref
            collect_DB_names = self.collect_DB_names
            new_Iter_db_dir = db_dir +'/data_%d' % (iteration+1)
            new_datanames = []

            for i in range(len(collect_DB_names)):
                db_name = collect_DB_names[i]
                new_dataname = db_name[:-3]+"_ref.data"
                if not os.path.exists(new_dataname):
                    new_dataname2 = reaxFFref_DFT_db(db_name, new_Iter_db_dir, reaxff_ref, lmp_path)
                    #print(new_dataname, new_dataname2)
                    print("Generate %s" % new_dataname2.split('/')[-1])
                new_datanames.append(new_dataname)
                

            return new_datanames
    
#Add function to collect QE results

def use_old_weight(training_dir, use_old_flag=True):
    os.chdir(training_dir)
    with open('input.nn','r') as inp:
        lines = inp.readlines()
        check = [True if 'use_old_weights_short' in line else False for line in lines]
        index = check.index(True)
        if use_old_flag:
            new_use_old = 'use_old_weights_short\n'
            lines[index] = new_use_old
        else:
            new_use_old = '#use_old_weights_short\n'
            lines[index] = new_use_old
    with open('input.nn','w') as inp2:
        for line in lines:
            inp2.write(line)

    return

def collect_nnp_data(Iter_training_dir):
    os.chdir(Iter_training_dir)
    nnp_data_dir = Iter_training_dir+'/nnp-data'
    if os.path.isdir(nnp_data_dir):
        weights_list = glob.glob('./nnp-data/weights.*.data')
        input_file = glob.glob('./nnp-data/input.nn')
        scaling_file = glob.glob('./nnp-data/scaling.data')
        if len(weights_list) > 0 and len(input_file) == 1 and len(scaling_file) == 1:
            return nnp_data_dir

    else:
        os.mkdir('./nnp-data')
        subprocess.call('cp input.nn ./nnp-data/input.nn',shell=True)
        subprocess.call('cp scaling.data ./nnp-data/scaling.data',shell=True)
        with open('input.nn','r') as f:
            contents = f.readlines()
            for line in contents:
                line2 = line.split()
                if len(line2) > 0 and line2[0] == 'number_of_elements':
                    nelements = int(line2[1])
                if len(line2) > 0 and line2[0] == 'elements':
                    elements = line2[1:1+nelements]

        for i in range(len(elements)):
            atomic_number = atomic_numbers[elements[i]]
            if atomic_number < 10:
                an = "00" + str(atomic_number)
            elif atomic_number >= 10 and atomic_number < 100:
                an = "0" + str(atomic_number)
            weights = glob.glob('weights.'+an+'*')
            sorted_w = sorted(weights)
            if 'weights.'+an+'.data' in sorted_w:
                sorted_w.remove('weights.'+an+'.data')
            weights2 = sorted_w[-1]
            weights2_data = weights2[:-10] + 'data'
            print(weights2, weights2_data)
            subprocess.call('cp '+weights2+' nnp-data/'+weights2_data,shell=True)

        return nnp_data_dir

def Launch_training(Iter_training_dir, queue):
    os.chdir(Iter_training_dir)
    nodes = queue['nodes']
    walltime = queue['walltime']
    input_nn = 'input.nn'
    input_data = 'input.data'
    if not os.path.exists('output.nn'):
        cmd = 'submit -n %d -w %s -i %s -i %s n2p2-c7b4407_nnp-norm_mpi' % (nodes,walltime,input_nn,input_data)
        subprocess.call(cmd,shell=True)
        subprocess.call('cp output.nn input.nn',shell=True)
    if not os.path.exists('scaling.data'):
        cmd = 'submit -n %d -w %s -i %s -i %s n2p2-c7b4407_nnp-scaling_mpi 100' % (nodes,walltime,input_nn,input_data)
        subprocess.call(cmd,shell=True)
    if not os.path.exists('learning-curve.out'):
        scaling_data = 'scaling.data'
        cmd = 'submit -n %d -w %s -i %s -i %s -i %s n2p2-c7b4407_nnp-train_mpi' % (nodes,walltime,input_nn,input_data,scaling_data)
        subprocess.call(cmd,shell=True)
    
    return

def Launch_dynamics(md_dir, queue):
    os.chdir(md_dir)
    nodes = queue['nodes']
    walltime = queue['walltime']
    input_files = '-i nnp-data -i ffield -i *.data'
    cmd = "submit -n %d -w %s %s lammps-03Mar20-parallel -i 'nnp.in'" % (nodes,walltime,input_files)
    subprocess.call(cmd,shell=True)
    print('submit calculation')
    
    return

def Launch_QE(dft_dir, qe_input, queue):
    os.chdir(dft_dir)
    pot_files = glob.glob('*.UPF')
    pot_files_ = [" -i "+p.split('/')[-1] for p in pot_files]
    pot_line = "".join(pot_files_)
    input_file = "-i "+qe_input
    nodes = queue['nodes']
    walltime = queue['walltime']
    cmd = 'submit -n %d -w %s %s espresso-6.2.1_pw %s' % (nodes,walltime,pot_line,input_file)
    subprocess.call(cmd,shell=True)

def check_qe_out(out_log):
    
    log = open(out_log,'r')
    log_lines = log.readlines()
    flags = []
    for i in range(len(log_lines)):
        if 'JOB DONE.' in log_lines[i]:
            flags.append(True)
        else:
            flags.append(False)
    check = flags.count(True)
    if check == 1:
        return True
    else:
        return False

def check_lmp_out(out_log):

    log = open(out_log,'r')
    log_lines = log.readlines()
    flags = []
    for i in range(len(log_lines)):
        if 'Total wall time:' in log_lines[i]:
            flags.append(True)
        else:
            flags.append(False)
    check = flags.count(True)
    if check > 0:
        return True
    else:
        return False

def write_qe_input(ase_atoms,qe_input):
    input_file = open(qe_input,'r')
    input_lines = input_file.readlines()
    #change the number of atoms
    for i in range(len(input_lines)):
        line = input_lines[i]
        if 'nat' in line:
            natoms_line = line.split('=')[0]
            new_natoms_line = natoms_line+'= %d\n' % len(ase_atoms)
            input_lines[i] = new_natoms_line

    #add positions and lattice parameter sections
    symbols = ase_atoms.get_chemical_symbols()
    positions = ase_atoms.get_positions()
    cell = ase_atoms.get_cell()

    cell_string = "\nCELL_PARAMETERS angstrom\n"
    pos_string = "\nATOMIC_POSITIONS alat\n"
    pos_line = ["%s\t %12.10f\t %12.10f\t %12.10f\n" % (symbols[i],positions[i][0],positions[i][1],positions[i][2]) for i in range(len(positions))]
    cell_line = ["%12.10f\t %12.10f\t %12.10f\n" % (cell[i][0],cell[i][1],cell[i][2]) for i in range(len(cell))]
    
    pos_string += "".join(pos_line)
    cell_string += "".join(cell_line)
    new_input_string = "".join(input_lines)
    new_input = new_input_string + pos_string + cell_string
    with open(qe_input,'w') as f:
        f.write(new_input)
    
    return
    

def QE_md_result(dft_spdir,output_file):
    #Collect geometry, energy, and forces
    #Write a single asedb file
    os.chdir(dft_spdir)
    check = glob.glob("*.stdout")
    output = open(dft_spdir+'/'+output_file,'r')
    lines = output.readlines()
    natoms_line = [line.split() for line in lines if 'number of atoms/cell' in line]
    natoms = int(natoms_line[0][-1])
    total_energy = 0
    forces_array = np.zeros((natoms,3))

    for l in range(len(lines)):
        if "!    total energy" in lines[l]:
            total_energy = float(lines[l].split()[-2])*13.6056980659
        if "Forces acting on atoms" in lines[l]:
            force_start_i = l
        elif "Total force =" in lines[l]:
            force_end_i = l
            forces = lines[force_start_i+2:force_end_i]
            forces_ = [f.split() for f in forces if len(f.split()) > 0]
            forces_ = np.array([f for f in forces_ if f[0] == 'atom'])
            #print(forces_)
            atom_ids = forces_[:,1].astype(int)
            x = forces_[:,6].astype(float)*25.71104309541616
            y = forces_[:,7].astype(float)*25.71104309541616
            z = forces_[:,8].astype(float)*25.71104309541616
            
            for ai in range(len(atom_ids)):
                index = atom_ids[ai] - 1
                forces_array[index][0] += x[ai]
                forces_array[index][1] += y[ai]
                forces_array[index][2] += z[ai]
    atoms = read('in_structure.xyz')
    atoms.set_calculator(SPC(atoms, energy=total_energy,forces=forces_array))
    
    return atoms



def QE_sp_result(dft_spdir,output_file):
    #Collect geometry, energy, and forces
    #Write a single asedb file
    os.chdir(dft_spdir)
    check = glob.glob("*.stdout")
    output = open(dft_spdir+'/'+output_file,'r')
    lines = output.readlines()
    natoms_line = [line.split() for line in lines if 'number of atoms/cell' in line]
    natoms = int(natoms_line[0][-1])
    forces_array = np.zeros((natoms,3))

    for l in range(len(lines)):
        if "!    total energy" in lines[l]:
            total_energy = float(lines[l].split()[-2])*13.6056980659
        if "Forces acting on atoms" in lines[l]:
            force_start_i = l
        elif "Total force =" in lines[l]:
            force_end_i = l
            forces = lines[force_start_i+2:force_end_i]
            forces_ = [f.split() for f in forces if len(f.split()) > 0]
            forces_ = np.array([f for f in forces_ if f[0] == 'atom'])
            #print(forces_)
            atom_ids = forces_[:,1].astype(int)
            x = forces_[:,6].astype(float)*25.71104309541616
            y = forces_[:,7].astype(float)*25.71104309541616
            z = forces_[:,8].astype(float)*25.71104309541616
            
            for ai in range(len(atom_ids)):
                index = atom_ids[ai] - 1
                forces_array[index][0] += x[ai]
                forces_array[index][1] += y[ai]
                forces_array[index][2] += z[ai]
    atoms = read('structure.xyz')
    atoms.set_calculator(SPC(atoms, energy=total_energy,forces=forces_array))
    
    return atoms


def reaxFFref_DFT_db(dbname, db_dir, reaxff_ref, lmp_path):
    os.chdir(db_dir)
    db = connect(dbname)
    images = list(db.select())
    new_dataname = dbname[:-3]+'_ref.data'
    new_images = []
    if not os.path.exists(new_dataname):
        
        s1 = []
        for i in range(len(images)):
            Nelements = len(images[i].positions)
            positions = images[i].positions
            energy = images[i].energy * (eV/ (kcal/mol) )
            force = images[i].forces
            cell = images[i].cell
            pbc = [1,1,1]
            cell_string = " ".join([str(c[0])+" "+str(c[1])+" "+str(c[2]) for c in cell])
            cell_string = '"' + cell_string +'"'

            elements = images[i].symbols
            counter = collections.Counter(elements)
            element_short = list(dict.fromkeys(elements))
            element_ordered = list(sorted(element_short, key=lambda e:atomic_numbers[e]))
            chargefile('charge.txt',element_ordered) #call chargefile func
            input_file('base.in',element_ordered,reaxff_ref)
            numbers = [ atomic_numbers[e] for e in elements ]

            ref_ref = 0
            dft_ref = 0

            elements_types = list(set(elements))
            elements_ordered = []
            positions_ordered = []
            forces_ordered = []
            for types in elements_types:
                for j in range(len(elements)):
                    if elements[j] == types:
                        e = elements[j]
                        elements_ordered.append(e)
                        p = positions[j]
                        positions_ordered.append(p)
                        p_string = ["%14.6f" % pp for pp in p]
                        p_s = "".join(p_string)
                        f = force[j]*(eV/(kcal/mol))
                        forces_ordered.append(f)
                        n = numbers[j]
                        f_string = ["%14.6f" % ff for ff in f]
                        f_s = "".join(f_string)
            formula = ""
            for types in elements_types:
                formula += types + str( elements_ordered.count(types) )
            atoms = Atoms(formula, cell=cell, pbc=pbc, positions=positions_ordered)
            chem = atoms.get_chemical_symbols()
            elements = list(set(chem))
            atom_types = {}
            for el, j in zip(elements, range(len(elements))):
                atom_types[el] = j+1

            basename = str(i+1)
            xyz_filename = basename + '.reax'
            write_lammps_data(filename=xyz_filename, atoms=atoms, atom_types=atom_types, units='real')
            lammps(basename, lmp_path, 1)

            #read forces
            ref_force = np.loadtxt('force.dump', skiprows=9)
            dft_force = np.array([list(f) for f in forces_ordered])
             
            #read energy
            dft_en = energy
            out = open('lmp.out', 'r')
            subprocess.call('cp lmp.out lmp_%d.out' % (i+1), shell=True)
            lines = out.readlines()
            for m in range(len(lines)):
                line = lines[m].split()
                if line[0] == 'Step':
                    index = line.index('PotEng')
                    thermout = lines[m+1].split()
                    ref_en = float(thermout[index])
                    index = line.index('Pxx')
                    press = thermout[index:]
                    ref_press = np.array([float(p) for p in press]) * 0.00010132501
                    break

            #dft_press = stress
            #DFT - REAX
            en = (dft_en-dft_ref) - (ref_en-ref_ref)
            factor = (kcal/mol/eV)
            en = en * factor
            #press = np.array(dft_press) - np.array(ref_press)
            force = [np.array(dft_force[i])*factor-np.array(ref_force[i])*factor for i in range(len(dft_force))]
            new_atoms = Atoms(formula, pbc=pbc,cell=cell,positions=positions_ordered)
            new_atoms.set_calculator(SPC(new_atoms,energy=en,forces=force))
            new_images.append(new_atoms)

        string = ""
        for new_image in new_images:
            string += atoms2data(new_image)
        new_data = open(new_dataname,'w')
        new_data.write(string)
        new_data.close()
        subprocess.call('rm lmp_* *.reax force.dump lmp.out min.xyz', shell=True)

        return new_dataname

def atoms2data(atoms):
    singledata = ""
    energy = atoms.get_potential_energy() * ( eV / (kcal/mol) )
    positions = atoms.get_positions()
    symbols= atoms.get_chemical_symbols()
    forces = atoms.get_forces() * ( eV / (kcal/mol) )
    cell = atoms.get_cell()

    singledata += "begin\n"
    singledata += "comment generated by PY\n"
    singledata += "lattice %20.10f %20.10f %20.10f\n" % (cell[0][0],cell[0][1],cell[0][2])
    singledata += "lattice %20.10f %20.10f %20.10f\n" % (cell[1][0],cell[1][1],cell[1][2])
    singledata += "lattice %20.10f %20.10f %20.10f\n" % (cell[2][0],cell[2][1],cell[2][2])

    for ai in range(len(symbols)):
        e = symbols[ai]
        p = positions[ai]
        f = forces[ai]
        p_string = ["%20.10f" % pp for pp in p]
        f_string = ["%20.10f" % ff for ff in f]
        p_s = "".join(p_string)
        f_s = "".join(f_string)
        singledata += "atom %s %s %20.10f %20.10f %s\n" %(p_s, e, 0.0, 0.0, f_s)

    singledata += "energy %20.10f\n" % energy
    singledata += "charge 0\n"
    singledata += "end\n"

    return singledata

def chargefile(filename, elements):
    with open(filename, 'w') as c1:
        c1.write('charge\n')
        elements = list(set(elements))
        for element in elements:
            c1.write(element + ' 0.0\n')

def lammps(lmp_filename,lmp_path, ncores):
    lammps_command ="mpiexec -np "+str(ncores)+" " + lmp_path + " -var filename " + lmp_filename + " -in base.in > lmp.out"
    subprocess.call(lammps_command, shell=True)
    return

def input_file(filename, elements, ffield):
    e_string = " ".join(elements)
    #print(e_string)

    with open(filename,'w') as ff:
        string =  "units           real\n"
        string += "boundary        p p p\n"
        string += "atom_style      charge\n"
        string += "neighbor        2.0 nsq\n"
        string += "neigh_modify    delay 2\n"
        string += "box tilt large\n"
        string += "read_data       ${filename}.reax\n\n"

        string += "#    Potential Parameters\n"
        string += "pair_style      reax/c NULL  safezone 4.0 mincap 400\n"
        string += "pair_coeff      * * "+ffield +" " + e_string + "\n"
        string += "compute reax all pair reax/c\n"
        string += "fix             10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n"
        string += "variable eb equal c_reax[1]\n"
        string += "variable ea equal c_reax[2]\n"
        string += "variable elp equal c_reax[3]\n"
        string += "variable emol equal c_reax[4]\n"
        string += "variable ev equal c_reax[5]\n"
        string += "variable epen equal c_reax[6]\n"
        string += "variable ecoa equal c_reax[7]\n"
        string += "variable ehb equal c_reax[8]\n"
        string += "variable et equal c_reax[9]\n"
        string += "variable eco equal c_reax[10]\n"
        string += "variable ew equal c_reax[11]\n"
        string += "variable ep equal c_reax[12]\n"
        string += "variable efi equal c_reax[13]\n"
        string += "variable eqeq equal c_reax[14]\n"
        string += "thermo_style    custom step v_eb v_ea v_elp v_emol v_ev v_epen v_ecoa v_ehb v_et v_eco v_ew v_ep v_efi v_eqeq pe pxx pyy pzz pxy pxz pyz\n"
        string += "thermo   1\n"
        string += "dump            d1 all custom 1 force.dump fx fy fz\n"
        string += "dump_modify     d1 sort id\n"
        string += "run           0\n"
        ff.write(string)

    return

def Dynamics_species_nanoHUB(trajectory,working_dir,
                             ffield_reax,ffoption,
                             interval,exclude,lmp_path,
                             rdict, ReaxFFBO):
    E = {}
    if ReaxFFBO:
        elements = reaxFF_botable(trajectory, working_dir, interval, ffield_reax, ffoption, lmp_path)
        for i, e in enumerate(elements):
            E[str(i+1)] = e
    else:
        elements = Rcutoff_botable(trajectory, working_dir, interval, rdict)
        for i, e in enumerate(elements):
            E[str(i+1)] = e
    output = Species_Rec('bonds.reaxc', 'dump_reax.dump',E,'data.in', exclude)

    return output

def reaxFF_botable(trajectory, working_dir, interval, ffield, ffoption, lmp_path):
    
    os.chdir(working_dir)
    images = read(trajectory,index=':')
    bo_string  = ""
    dump_string = ""
    s1 = []
    for i in range(0,len(images),interval):
        atoms = images[i]
        chem = atoms.get_chemical_symbols()
        elements = sorted(list(set(chem)), key=lambda x: atomic_numbers[x])
        atom_types = {}
        for el, j in zip(elements, range(len(elements))):
            atom_types[el] = j+1
        chargefile('charge.txt', elements)
        input_file_reaxFFBO('base.in',elements,ffield,ffoption)
        basename = str(i+1)
        xyz_filename = basename + '.reax'
        write_lammps_data(filename=xyz_filename, atoms=atoms, atom_types=atom_types, units='real')
        if i == 0:
             write_lammps_data(filename='data.in', atoms=atoms, atom_types=atom_types, units='real')
        lammps(basename, lmp_path, 1)

        #read bo file of each image
        bo_out = 'bonds.reaxc_%d' % (i+1)
        dump_out = 'min.dump_%d' % (i+1)
        md_out = 'lmp.out'

        r_bo = open(bo_out, 'r').readlines()
        t1 = r_bo[0].split()
        t1[2] = str(i+1) + "\n"
        r_bo[0] = " ".join(t1)
        r_dump = open(dump_out).readlines()
        r_dump[1] = str(i+1) + "\n"

        bo_string += "".join(r_bo)
        dump_string += " ".join(r_dump)

    with open('bonds.reaxc','w') as f1:
        f1.write(bo_string)
    with open('dump_reax.dump','w') as f2:
        f2.write(dump_string)

    subprocess.call('rm *.lmp *.reax force.dump lmp.out min.dump_* bonds.reaxc_*', shell=True)
    return elements


def input_file_reaxFFBO(filename, elements, ffield, ffoption=''):
    elements0 = sorted(elements,key=lambda x: atomic_numbers[x])
    e_string = " ".join(elements0)
    with open('lmp_control','w') as c:
        string0 = "simulation_name      ReaxFF ! output files will carry this name + their specific ext\n"
        string0 += "tabulate_long_range 10000 ! denotes the granularity of long range tabulation, 0 means no tabulation\n"
        string0 += "energy_update_freq  1\n"
        string0 += "nbrhood_cutoff      5.0  ! near neighbors cutoff for bond calculations in A\n"
        string0 += "hbond_cutoff        6.0  ! cutoff distance for hydrogen bond interactions 6.0 default\n"
        string0 += "bond_graph_cutoff   0.3  ! bond strength cutoff for bond graphs\n"
        string0 += "thb_cutoff      0.001 ! cutoff value for three body interactions\n"
        string0 += "write_freq      100000    ! write trajectory after so many steps\n"
        string0 += "traj_title      RDX_ReaxFF ! (no white spaces)\n"
        string0 += "atom_info       0    ! 0: no atom info, 1: print basic atom info in the trajectory file\n"
        string0 += "atom_forces     0    ! 0: basic atom format, 1: print force on each atom in the trajectory file\n"
        string0 += "atom_velocities     0    ! 0: basic atom format, 1: print the velocity of each atom in the trajectory file\n"
        string0 += "bond_info       0    ! 0: do not print bonds, 1: print bonds in the trajectory file\n"
        string0 += "angle_info      0    ! 0: do not print angles, 1: print angles in the trajectory file\n"
        c.write(string0)

    with open(filename,'w') as ff:
        string =  "units           real\n"
        string += "boundary        p p p\n"
        string += "atom_style      charge\n"
        string += "neighbor        2.0 nsq\n"
        string += "neigh_modify    delay 2\n"
        string += "box             tilt large\n"
        string += "read_data       ${filename}.reax\n\n"

        string += "#    Potential Parameters\n"
        string += "pair_style      reax/c lmp_control safezone 4.0 mincap 400 "+ ffoption +"\n"
        string += "pair_coeff      * * "+ffield +" " + e_string + "\n"
        string += "compute reax all pair reax/c\n"
        string += "fix             10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n"
        string += "variable eb equal c_reax[1]\n"

        string += "variable ea equal c_reax[2]\n"
        string += "variable elp equal c_reax[3]\n"
        string += "variable emol equal c_reax[4]\n"
        string += "variable ev equal c_reax[5]\n"
        string += "variable epen equal c_reax[6]\n"
        string += "variable ecoa equal c_reax[7]\n"
        string += "variable ehb equal c_reax[8]\n"
        string += "variable et equal c_reax[9]\n"
        string += "variable eco equal c_reax[10]\n"
        string += "variable ew equal c_reax[11]\n"
        string += "variable ep equal c_reax[12]\n"
        string += "variable efi equal c_reax[13]\n"
        string += "variable eqeq equal c_reax[14]\n"
        string += "thermo_style    custom step v_eb v_ea v_elp v_emol v_ev v_epen v_ecoa v_ehb v_et v_eco v_ew v_ep v_efi v_eqeq pe pxx pyy pzz pxy pxz pyz\n"
        string += "thermo   1\n"

        string += "fix             rbo all reax/c/bonds 2 bonds.reaxc_${filename}\n"
        string += "dump            min all custom 2 min.dump_${filename} id type x y z q fx fy fz\n"
        string += "dump            d1 all custom 1 force.dump fx fy fz\n"
        string += "dump_modify     d1 sort id\n"
        string += "min_style       cg\n"
        string += "minimize        1e-8 1e-8  0  0\n"
        string += "run             1\n"
        ff.write(string)

    return
