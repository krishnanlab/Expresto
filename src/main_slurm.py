import numpy as np
import pandas as pd
import argparse
import time
import glob
import os
from subprocess import Popen, PIPE
from itertools import combinations, product


parser = argparse.ArgumentParser()
parser.add_argument('-m','--model',
                    default = 'GeneLasso',
                    type = str,
                    help = 'GeneLasso, SampleLasso, SampleKNN, GeneKNN')
parser.add_argument('-hp','--hyperparameter',
                    default = 0.01,
                    type = float,
                    help = 'with k or alpha (k will be turned into an int)')
parser.add_argument('-trnd','--trndata',
                    default = 'Microarray',
                    type = str,
                    help = 'Microarray, RNAseq')
parser.add_argument('-tstd','--tstdata',
                    default = 'Microarray',
                    type = str,
                    help = 'Microarray, RNAseq, or RNAseq-Microarray')
parser.add_argument('-s','--split',
                    default = 'Val',
                    type = str,
                    help = 'Val, Tst (always uses trimmed val)')
parser.add_argument('-gs','--genesplit',
                    default = 'LINCS',
                    type = str,
                    help = 'GPL96-570, LINCS')
parser.add_argument('-b','--betas',
                    default = 'yes',
                    type = str,
                    help = 'yes, no (Whether to sample LASSO betas or neighbors in KNN)')
parser.add_argument('-sd','--savedir',
                    default = '../reproduce_results/LASSO-KNN-save/',
                    type = str,
                    help = 'The base dir where all the specific directories are')
parser.add_argument('-mem','--memory',
                    default = '50G',
                    type = str,
                    help = 'Memory for the job')
parser.add_argument('-nm','--nummodels',
                    default = 100,
                    type = int,
                    help = 'The number of models to do for each job (only for LASSO)')
parser.add_argument('-r','--runtime',
                    default = '03:50:00',
                    type = str,
                    help = 'The runtime for each job')
args = parser.parse_args()
Model = args.model
HyperParameter = args.hyperparameter
TrnData = args.trndata
TestData = args.tstdata
Split = args.split
GeneSplit = args.genesplit
Betas = args.betas
SaveBaseDir = args.savedir
Memory = args.memory
NumModels = args.nummodels
RunTime = args.runtime


# dir to put the slurm files
slurm_dir = '../reproduce_results/slurms/'
# dir where final results are transfered to
# This was used to move data out of temporary scratch drive
# so not really used here
results_dir = SaveBaseDir

# start to measure how long it takes the script to run
tic = time.time()

# get the TestData values to make multiple folders if needed
TestData_tmp = TestData.strip().split('-')


# check the parameters to see if the folders needed exisits
# if not then create them
save_dirs_dict = {}
for datatype in TestData_tmp:
    save_dir_end = 'M--%s__H--%s__Trn--%s__Tst--%s__S--%s__GS--%s/'%(Model,HyperParameter,TrnData,
                                                                     datatype,Split,GeneSplit)
    save_dir = SaveBaseDir + save_dir_end
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dirs_dict[datatype] = save_dir

# check to see if all KNN files trying to create are already done
if Model in ['SampleKNN','GeneKNN']:
    issame = 0
    for akey in save_dirs_dict:
        # get FNs done in scratch space
        FNs = glob.glob(save_dirs_dict[akey]+'*npy')
        FNs = [aFN.strip().split('/')[-1] for aFN in FNs]
        # get FNs done in compbio space
        results_dir_tmp = results_dir + save_dirs_dict[akey].strip().split('/')[-2] + '/'
        FNs_compbio = glob.glob(results_dir_tmp +'*npy')
        FNs_compbio = [aFN.strip().split('/')[-1] for aFN in FNs_compbio]
        if Betas == 'no':
            files_ = ['preds.npy']
        else:
            files_ = ['preds.npy','dists.npy','inds.npy']
        if (len(files_) == len(np.intersect1d(files_,FNs))) or (len(files_) == len(np.intersect1d(files_,FNs_compbio))):
            issame = issame + 1
    if issame == len(save_dirs_dict):
        raise FileExistsError('All the KNN files have already been made')
    else:
        print('Some file not done, so going to run the KNN models')
        ModelInds_list = ['NoInds'] # this will be a list of one empty string
        

# for Lasso check to see what models need to be run still
# data_dir = '../data/'
data_dir = '../data/'
if Model in ['SampleLasso','GeneLasso']:    
    if Model == 'GeneLasso':
        good_inds = len(np.loadtxt(data_dir+'%s_ygenes_inds.txt'%GeneSplit,dtype=int))
    if Model =='SampleLasso':
        if Split == 'Tst':
            good_inds = np.load(data_dir+'%s_%s_Exp.npy'%(TestData_tmp[0],Split),mmap_mode='r').shape[0]
        if Split == 'Val':
            good_inds = len(np.load(data_dir+'%s_trimmed_Val_inds.npy'%TestData_tmp[0]))
    good_inds = np.arange(good_inds)
    done_in_both_preds = []
    done_in_both_betas = []
    issame = 0
    for akey in save_dirs_dict:
        # get FNs done in scratch space
        FNs = glob.glob(save_dirs_dict[akey]+'*npy')
        FNs = [aFN.strip().split('/')[-1] for aFN in FNs]
        # get FNs done in compbio space
        results_dir_tmp = results_dir + save_dirs_dict[akey].strip().split('/')[-2] + '/'
        FNs_compbio = glob.glob(results_dir_tmp +'*npy')
        FNs_compbio = [aFN.strip().split('/')[-1] for aFN in FNs_compbio]
        if Betas == 'no':
            files_ = ['preds.npy']
        else:
            files_ = ['preds.npy','betas.npy']
        if (len(files_) == len(np.intersect1d(files_,FNs))) or (len(files_) == len(np.intersect1d(files_,FNs_compbio))):
            issame = issame + 1
        FNs_pred = glob.glob(save_dirs_dict[akey]+'preds__ModelInds*npy')
        FNs_beta = glob.glob(save_dirs_dict[akey]+'betas__ModelInds*npy')
        pred_inds = [int(item.split('.npy')[0].split('--')[-1]) for item in FNs_pred]
        beta_inds = [int(item.split('.npy')[0].split('--')[-1]) for item in FNs_beta]
        done_in_both_preds.append(pred_inds)
        done_in_both_betas.append(beta_inds)
    if len(save_dirs_dict) == 2:
        done_in_both_preds = np.intersect1d(done_in_both_preds[0],done_in_both_preds[1])
        done_in_both_betas = np.intersect1d(done_in_both_betas[0],done_in_both_betas[1])
    elif len(save_dirs_dict) == 1:
        done_in_both_preds = np.array(done_in_both_preds)
        done_in_both_betas = np.array(done_in_both_betas)
    else:
        raise FileExistsError('Not a good number of test sets to try')
    if issame == len(save_dirs_dict):
        raise FileExistsError('All the Lasso pred and beta files have already been made')
    if Betas == 'no':
        inds_to_do = np.setdiff1d(good_inds,done_in_both_preds)
    if Betas == 'yes':
        done_in_both_all = np.intersect1d(done_in_both_preds,done_in_both_betas)
        inds_to_do = np.setdiff1d(good_inds,done_in_both_all)
    if len(inds_to_do) == 0:
        raise ValueError('All Lasso ModelInd files have been run and saved for this parameter set')
    # now make the splits or if empty raise an error
    ModelInds_list = []
    for idx in range(800): #
        start_ind = idx * NumModels
        stop_ind = (idx * NumModels) + NumModels
        if stop_ind > len(inds_to_do):
            model_tmp = inds_to_do[start_ind:stop_ind]
            model_tmp = ','.join([str(item) for item in model_tmp])
            ModelInds_list.append(model_tmp)
            break
        else:
            model_tmp = inds_to_do[start_ind:stop_ind]
            model_tmp = ','.join([str(item) for item in model_tmp])
            ModelInds_list.append(model_tmp)
    print('The number of Lasso job to start is',len(ModelInds_list))
        

# make the file_name
slurm_FN = 'M--%s__H--%s__Trn--%s__Tst--%s__S--%s__GS--%s'%(Model,HyperParameter,TrnData,TestData,Split,GeneSplit)
print('Some job parameters are',slurm_FN)


for idx, aModelInd_set in enumerate(ModelInds_list):

    mylist = ['#!/bin/bash']
    mylist.append('### define resources needed:')
    mylist.append('#SBATCH --time=%s'%RunTime)
    mylist.append('#SBATCH --nodes=1')
    mylist.append('#SBATCH --mem=%s'%Memory)
    mylist.append('#SBATCH --cpus-per-task=1')
    mylist.append('#SBATCH --job-name=Job-%s__%i'%(slurm_FN,idx))
    mylist.append('#SBATCH --output=%sslurm-%%x-%%j.out'%slurm_dir)
    mylist.append('umask g+rw')
    mylist.append('cd ../src')
    mylist.append('python main.py -m %s -hp %s -trnd %s -tstd %s -s %s -gs %s -b %s -mi %s -sd %s'%(Model,
                    HyperParameter, TrnData, TestData, Split, GeneSplit, Betas, aModelInd_set, SaveBaseDir))

print('This script took %i minutes to run '%((time.time()-tic)/60))
