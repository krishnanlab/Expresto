import os
import time
import argparse
import numpy as np
import glob
import itertools


'''
make it so directory for data is made if not there, use same name as in main
remove the if statement for rerunning the jobs
'''

# start to measure how long it takes the script to run
tic = time.time()

# define some dirs
slurm_dir = '../reproduce_results/slurms/'
data_dir = '../data/'


# load LINCS y-genes
ygenes = np.loadtxt(data_dir+'LINCS_ygenes_inds.txt',dtype=int)
num_samps = int(np.rint(len(ygenes)/4))
samp_inds = [[0,num_samps],[num_samps,2*num_samps],[2*num_samps,3*num_samps],[3*num_samps,len(ygenes)]]

data_splits = [['Microarray','Microarray'],['RNAseq','Microarray'],['RNAseq','RNAseq']]

models = []
models.append(['adadelta'])
models.append(['adam','0.01'])
models.append(['adam','0.001'])
models.append(['adam','0.0001'])
models.append(['adam','0.00001'])
models.append(['adam','0.000001'])

# add inds to the models
final_models_tmp = []
for amodel in models:
    for anind in samp_inds:
        final_models_tmp.append(amodel + anind)

final_models = []
for amodel in final_models_tmp:
    for asplit in data_splits:
        final_models.append(asplit + amodel)

print(len(final_models))
print(final_models)
        

for idx, param in enumerate(final_models):

    if idx in [47,59,62]:

        mylist = ['#!/bin/bash']
        mylist.append('### define resources needed:')
        mylist.append('#SBATCH --time=20:00:00')
        mylist.append('#SBATCH --gres=gpu:k80:1')
        mylist.append('#SBATCH --mem=100G')
        mylist.append('#SBATCH --job-name=DNN-%s'%idx)
        mylist.append('#SBATCH --output='+slurm_dir+'slurm-%x-%j.out')
        mylist.append('umask g+rw')
        mylist.append('cd ../src')
        if param[2] == 'sgd':
            mylist.append('python DNN_main.py -trn %s -tst %s -stm %s -spm %s -opt %s -lr %s -m %s'%(param[0],param[1],param[-2],param[-1],
                                                                                                     param[2],param[3],param[4]))
        elif param[2] == 'adam':
            mylist.append('python DNN_main.py -trn %s -tst %s -stm %s -spm %s -opt %s -lr %s'%(param[0],param[1],param[-2],param[-1],
                                                                                               param[2],param[3]))
        elif param[2] == 'adadelta':
            mylist.append('python DNN_main.py -trn %s -tst %s -stm %s -spm %s -opt %s'%(param[0],param[1],param[-2],param[-1],
                                                                                        param[2]))
        else:
            print('Not a valid optimizer')

        with open(slurm_dir+'%s.sb'%idx, 'w') as thefile:
            for item in mylist:
                thefile.write("%s\n" % item)

        os.system('sbatch %s%s.sb'%(slurm_dir,idx))


print('This script took %i minutes to run '%((time.time()-tic)/60))
