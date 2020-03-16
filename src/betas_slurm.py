import numpy as np
import time
import os

tic = time.time()

# select the number of model to do per job for Lasso
N = 2

# dir to put the slurm files
slurm_dir = '../reproduce_results/slurms/'

# get the number of Inds to do
data_dir = '../data/'
beta_tst_inds = np.loadtxt(data_dir + 'Beta_Tst_Inds.txt', dtype=int)
# find the len of beta_tst_inds
num_samps = len(beta_tst_inds)
# convert this to an ordered list from zero (real inds are used to slice data in betas_main.py)
samps_inds = list(range(num_samps))
# change int to str for using join later
samps_inds = [str(item) for item in samps_inds]
# make a comma separated elements in list
ModelInds_list = [','.join(samps_inds[n:n+N]) for n in range(0, len(samps_inds), N)]
print('The number of jobs to submit is',len(ModelInds_list))


for idx, aModelInd_set in enumerate(ModelInds_list):

    mylist = ['#!/bin/bash']
    mylist.append('### define resources needed:')
    mylist.append('#SBATCH --time=03:50:00')
    mylist.append('#SBATCH --nodes=1')
    mylist.append('#SBATCH --mem=100G')
    mylist.append('#SBATCH --cpus-per-task=1')
    mylist.append('#SBATCH --job-name=beta_analysis-%i'%idx)
    mylist.append('#SBATCH --output=%sslurm-%%x-%%j.out'%slurm_dir)
    mylist.append('umask g+rw')
    mylist.append('cd ../src')
    mylist.append('python betas_main.py -mi %s'%aModelInd_set)

    with open(slurm_dir + 'beta_analysis-%i.sb'%idx, 'w') as thefile:
        for item in mylist:
            thefile.write("%s\n" % item)

    os.system('sbatch ' + slurm_dir + 'beta_analysis-%i.sb'%idx)

print('This script took %i minutes to run '%((time.time()-tic)/60))
