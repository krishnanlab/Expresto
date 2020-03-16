#!/bin/bash

# python main_slurm.py -m SampleKNN -hp 20 -trnd Microarray -tstd Microarray -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00
# python main_slurm.py -m SampleKNN -hp 20 -trnd Microarray -tstd Microarray -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00
# python main_slurm.py -m SampleKNN -hp 5 -trnd RNAseq -tstd Microarray -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00
# python main_slurm.py -m SampleKNN -hp 1 -trnd RNAseq -tstd Microarray -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00
# python main_slurm.py -m SampleKNN -hp 20 -trnd RNAseq -tstd RNAseq -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00
# python main_slurm.py -m SampleKNN -hp 20 -trnd RNAseq -tstd RNAseq -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00

# python main_slurm.py -m GeneKNN -hp 5 -trnd Microarray -tstd Microarray -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00
# python main_slurm.py -m GeneKNN -hp 5 -trnd Microarray -tstd Microarray -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00
# python main_slurm.py -m GeneKNN -hp 20 -trnd RNAseq -tstd Microarray -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00
# python main_slurm.py -m GeneKNN -hp 20 -trnd RNAseq -tstd Microarray -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00
# python main_slurm.py -m GeneKNN -hp 20 -trnd RNAseq -tstd RNAseq -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00
# python main_slurm.py -m GeneKNN -hp 20 -trnd RNAseq -tstd RNAseq -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -r 03:50:00

# python main_slurm.py -m GeneLasso -hp 0.0001 -trnd Microarray -tstd Microarray -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 150 -r 03:50:00
# python main_slurm.py -m GeneLasso -hp 0.001 -trnd Microarray -tstd Microarray -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 10 -r 03:50:00
# python main_slurm.py -m GeneLasso -hp 0.0001 -trnd RNAseq -tstd Microarray -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 150 -r 03:50:00
# python main_slurm.py -m GeneLasso -hp 0.001 -trnd RNAseq -tstd Microarray -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 10 -r 03:50:00
# python main_slurm.py -m GeneLasso -hp 0.00001 -trnd RNAseq -tstd RNAseq -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 150 -r 03:50:00
# python main_slurm.py -m GeneLasso -hp 0.001 -trnd RNAseq -tstd RNAseq -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 10 -r 03:50:00

# python main_slurm.py -m SampleLasso -hp 0.01 -trnd Microarray -tstd Microarray -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 150 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 0.001 -trnd Microarray -tstd Microarray -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 2 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 0.1 -trnd RNAseq -tstd Microarray -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 150 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 0.01 -trnd RNAseq -tstd Microarray -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 2 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 0.1 -trnd RNAseq -tstd RNAseq -s Tst -gs LINCS -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 150 -r 03:50:00
# python main_slurm.py -m SampleLasso -hp 0.01 -trnd RNAseq -tstd RNAseq -s Tst -gs GPL96-570 -b yes -sd /mnt/gs18/scratch/groups/compbio/imputation/ -mem 100G -nm 2 -r 03:50:00
