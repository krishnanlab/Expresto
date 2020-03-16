# Expresto
This repository contains data and code to generate the results and reproduce the figures and tables found in [_A Flexible, Interpretable, and Accurate Approach for Imputing the Expression of Unmeasured Genes_](https://www.needtopostsomewhere.com) (submitted for review). This work introduces a new method for imputing gene expression. The method introduced, SampleLASSO, uses the LASSO machine learning algorithm in a way that captures context specific biologically relevant information to guide imputation. 

This repo provides: 
1. The data, results, and figures presented in the manuscript.
2. Code to regenerate the results and figures.
3. A function that allows a user to upload a dataset to be imputed, and then we use SampleLASSO to fill in the unmeasured genes and also report which other expression samples in the training data were the most helpful for imputation. 

## Section 1: Pre-computed Data, Results, and Figures/Tables
### Data
The data used in this study (networks, embeddings, and genesets) is available on [Zenodo](https://zenodo.org/record/3711089#.Xm7ZLJNKgWo). To get the data run
```
sh get_data.sh
```

### Results
`results/`: This directory contains three subdirectories:
1. `LASSO_KNN_results`
2. `DNN_results`
3. `Beta_Analysis_Results`

### Figures/Tables
`figures_tables/`: This directory contains all the pre-generated figures and tables.

## Section 2: Regenerating the Results and Figures/Tables
### Dependencies
This code was tested on an Anaconda distribution of python. The major packages used are:
```
python 3.7 
numpy 1.16.4
scipy 1.3.0
pandas 0.24.2
scikit-learn 0.20.3
matplotlib 3.0.3
seaborn 0.9.0
statsmodels 0.9.0
tensorflow-gpu 1.14.0 (this was run with python 3.6)
keras-gpu 2.2.4 (this was run with python 3.6)
```
The parallelization of the code was tested with Slurm on the high performance computing cluster at Michigan State University.

### Running LASSO and KNN code
1. `main.py`: Main script that generates imputed values
2. `main_utls.py`: Helper function for main.py
3. `main_slurm.py`: A python script that will submit numerous jobs through slurm
4. `run_GeneKNN_val_jobs.sh`, `run_GeneLasso_val_job.sh`, `run_SampleKNN_val_jobs.sh`, `run_SampleLasso_val_job.sh` are scripts that start running the relevant jobs.
5. `main_knitting`: Combines all predictions for one hyperparameter set into one file
6. `main_evalautions`: Makes a file that has evaluations for different metrics

### Running DNN code
1. `DNN_main.py`: Main script that generates imputed values, and makes the evaluation file
2. `DNN_slurm.py`: A python script that submits all relevant DNN jobs.

### Running Beta Analysis code
1. `beta_main`: Main script that generates imputed values
2. `betas_slurm`: A python script that submits the jobs through slurm
3. `betas_knitting_evals_move.py`: This combines all predictions for one hyperparameter set into one file and make a file for evaluations of different metrics 


## Section 3: User function for imputing any data
To impute an new data use the function found at `src/user_function.py` which as the following arguments
1. `-mgf, --measured_genes_file`: The path to a tab separated file where the rows are the different genes, the first column contains the gene IDs and the rest of the columns contain the expression data to be imputed.
2. `-t, --targets`: The path to a text file containing the gene IDs of unmeasrued genes that need to be imputed. If this path is not given, then all the genes in the training set that are not in the measured_genes_file will be imputed
3. `-td, --training_data`: The path to the data to be used for training (right now need to be a numpy array that has samples along the rows and genes along the columns)
4. `-id, --gene_ids`: The path to the file that maps the columns in the training data to gene IDs
5. `-tk, --training_key`: The path that maps the GSE and GSM IDs to the samples in the training set
6. `-upd, --use_all_paper_data`: If this argument is set to either **Microarray** or **RNAseq** the function will ignore arguments 3-5 and just use the pre-supplied data used this work.

An example to run is 
```
cd src
python user_function.py -mgf ../data/example_data.tsv -t ../data/example_targets.tsv -td ../data/Microarray_Trn_Exp.npy -id ../data/GeneIDs.txt -tk ../data/Microarray_Trn_Key.tsv
```
This function output 4 files into the directory `user_results` in a subdirectory that is label with the timestamp YYYY-MM-DD-HH-SS
1. `predictions.tsv`: A tab separated file with the first column being the Gene IDs and the rest of the columns being the imputed expression values
2. `top_betas.tsv`: A tab separated file where for each GSM that was imputed, it gives back 100 training samples with the highest model coefficients  
3. `unusable_measured_genes.txt`: A text file containing gene IDs in the uploaded measured_genes_file that were not in the training set
4. `unusable_targets.txt`: A text file that list gene IDs of target genes not imputed because they were also in the measured_genes_file


## Section 4: Additional Information
### Support
For support please contact Jake Canfield at canfie44@msu.edu or [Chris Mancuso](https://twitter.com/ChrisAMancuso) at mancus16@msu.edu.

### License
See [LICENSE.md](https://github.com/krishnanlab/SampleLasso-dev/LICENSE.md) for license information for all data used in this project.

### Citation
If you use this work, please cite:  
`To be added later`

### Authors
Christopher A Mancuso#, Jake Canfield#, Deepak Singla, Arjun Krishnan*

\# These authors are joint first authors
\* General correspondence should be addressed to AK at arjun@msu.edu.

### Funding
This work was primarily supported by US National Institutes of Health (NIH) grants R35 GM128765 to AK and in part by MSU start-up funds to AK and NIH F32 Fellowship SOMENUMBERHERE for CM.

### Acknowledgements
We are grateful for the support from the members of the [Krishnan Lab](https://www.thekrishnanlab.org).

### Referecnes

Down here cite ARCHS4 and GEO, maybe fRMA and custom CDF. 
