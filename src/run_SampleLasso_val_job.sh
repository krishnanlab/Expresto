#!/bin/bash


# python main_slurm.py -m SampleLasso -hp 0.0001 -trnd Microarray -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done
# python main_slurm.py -m SampleLasso -hp 0.001 -trnd Microarray -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done
# python main_slurm.py -m SampleLasso -hp 0.01 -trnd Microarray -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done
# python main_slurm.py -m SampleLasso -hp 0.1 -trnd Microarray -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done
# python main_slurm.py -m SampleLasso -hp 1 -trnd Microarray -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done
# python main_slurm.py -m SampleLasso -hp 10 -trnd Microarray -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done

# python main_slurm.py -m SampleLasso -hp 0.0001 -trnd Microarray -tstd Microarray -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 10 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 0.001 -trnd Microarray -tstd Microarray -s Val -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done with betas
# python main_slurm.py -m SampleLasso -hp 0.01 -trnd Microarray -tstd Microarray -s Val -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done with betas
# python main_slurm.py -m SampleLasso -hp 0.1 -trnd Microarray -tstd Microarray -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done with betas
# python main_slurm.py -m SampleLasso -hp 1 -trnd Microarray -tstd Microarray -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done
# python main_slurm.py -m SampleLasso -hp 10 -trnd Microarray -tstd Microarray -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00 # done

# python main_slurm.py -m SampleLasso -hp 0.0001 -trnd RNAseq -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 0.001 -trnd RNAseq -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 0.01 -trnd RNAseq -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 0.1 -trnd RNAseq -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r  03:50:00 # done with betas
# python main_slurm.py -m SampleLasso -hp 1 -trnd RNAseq -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 10 -trnd RNAseq -tstd Microarray -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00

# python main_slurm.py -m SampleLasso -hp 0.0001 -trnd RNAseq -tstd Microarray -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 3 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 0.001 -trnd RNAseq -tstd Microarray -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 3 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 0.01 -trnd RNAseq -tstd Microarray -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 0.1 -trnd RNAseq -tstd Microarray -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 1 -trnd RNAseq -tstd Microarray -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 10 -trnd RNAseq -tstd Microarray -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00

# python main_slurm.py -m SampleLasso -hp 0.0001 -trnd RNAseq -tstd RNAseq -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 0.001 -trnd RNAseq -tstd RNAseq -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 0.01 -trnd RNAseq -tstd RNAseq -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 0.1 -trnd RNAseq -tstd RNAseq -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 1 -trnd RNAseq -tstd RNAseq -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 10 -trnd RNAseq -tstd RNAseq -s Val -gs LINCS -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00

# python main_slurm.py -m SampleLasso -hp 0.0001 -trnd RNAseq -tstd RNAseq -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 80 -r 12:00:00
# python main_slurm.py -m SampleLasso -hp 0.001 -trnd RNAseq -tstd RNAseq -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 10 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 0.01 -trnd RNAseq -tstd RNAseq -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 0.1 -trnd RNAseq -tstd RNAseq -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 1 -trnd RNAseq -tstd RNAseq -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 10 -trnd RNAseq -tstd RNAseq -s Val -gs GPL96-570 -b no -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 20 -r 03:50:00