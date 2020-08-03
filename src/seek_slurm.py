import numpy as np
import time
import os
from subprocess import Popen, PIPE
import glob

tic = time.time()

# dir to put the slurm files
slurm_dir = '../reproduce_results/slurms/'
fp_seek = '../data/Multiple_Platforms/'


# get the different GPLs to try
GPLs = glob.glob(fp_seek + '*key.tsv')
GPLs = [item.split('/')[-1].split('_')[0] for item in GPLs]
print(GPLs)
print()

methods = ['SL','GL','DNN','GGAN']

combos = []
for amethod in methods:
    for aGPL in GPLs:
        combos.append([amethod,aGPL])
        
print(combos)


for idx, acombo in enumerate(combos):

    mylist = ['#!/bin/bash']
    mylist.append('### define resources needed:')
    if acombo[0] in ['SL','GL']:
        mylist.append('#SBATCH --time=20:00:00')
    elif acombo[0] in ['DNN']:
        mylist.append('#SBATCH --time=30:00:00')
    elif acombo[0] in ['GGAN']:
        mylist.append('#SBATCH --time=47:30:00')
    else:
        print('Not a good method')
        break
    if acombo[0] in ['SL','GL']:
        mylist.append('#SBATCH --nodes=1')
        mylist.append('#SBATCH --cpus-per-task=1')
    elif acombo[0] in ['DNN','GGAN']:
        mylist.append('#SBATCH --gres=gpu:k80:1')
    else:
        print('Not a good method')
        break
    mylist.append('#SBATCH --mem=100G')
    mylist.append('#SBATCH --job-name=%s-%s'%(acombo[0],acombo[1]))
    mylist.append('#SBATCH --output=%sslurm-%%x-%%j.out'%slurm_dir)
    mylist.append('python seek_%s.py -g %s'%(acombo[0],acombo[1]))

    with open(slurm_dir + '%s-%s.sb'%(acombo[0],acombo[1]), 'w') as thefile:
        for item in mylist:
            thefile.write("%s\n" % item)

    os.system('sbatch ' + slurm_dir + '%s-%s.sb'%(acombo[0],acombo[1]))

print('This script took %i minutes to run '%((time.time()-tic)/60))
