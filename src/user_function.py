import pandas as pd
import numpy as np
import argparse
import time
import os
from sklearn.preprocessing import StandardScaler # tested with verison 0.20.3
from sklearn.linear_model import Lasso # tested with verison 0.20.3
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('-mgf','--measured_genes_file',
                    default = '../data/example_data.tsv',
                    type = str,
                    help = 'Filepath to the data file')
parser.add_argument('-t','--targets',
                    default = 'None',
                    type = str,
                    help = 'Filepath to the targets file')
parser.add_argument('-td','--training_data',
                    default = '../data/Microarray_Trn_Exp.npy',
                    type = str,
                    help = 'Filepath to numpy array of data ')
parser.add_argument('-id','--gene_ids',
                    default = '../data/GeneIDs.txt',
                    type = str,
                    help = 'Filepath for the traning set gene IDs')
parser.add_argument('-tk','--training_key',
                    default = '../data/Microarray_Trn_Key.tsv',
                    type = str,
                    help = 'Filepath to a tsv with GSM and GSE data for training data' + 
                           'First column is GSM and second is GSE,needs column headers')
parser.add_argument('-upd','--use_all_paper_data',
                    default = 'None',
                    type = str,
                    help = 'If this is Microarray or RNAseq it will use' +
                           'all data (trn,val,test) from the paper and ignore' +
                           'the td, id and tk arguments')
args = parser.parse_args()

tic0 = time.time()
# get data to seconds for saving results
now = datetime.datetime.now()
rightnow = '%04d-%02d-%02d-%02d-%02d'%(now.year,now.month,now.day,now.hour,now.second)

# load user data
print('Loading files for measured genes, targets and training set gene IDs and key','\n')
df_data = pd.read_csv(args.measured_genes_file,sep='\t')
# load GeneIDs
if args.use_all_paper_data in ['Microarray','RNAseq']:
    GeneIDs = np.loadtxt('../data/GeneIDs.txt',dtype=str)
else:
    GeneIDs = np.loadtxt(args.gene_ids,dtype=str)
# load the data key
if args.use_all_paper_data in ['Microarray','RNAseq']:
    datatype = args.use_all_paper_data
    key_df = pd.DataFrame()
    for item in ['Trn','Val','Tst']:
        df_tmp = pd.read_csv('../data/%s_%s_Key.tsv'%(datatype,item),sep='\t')
        key_df = pd.concat([key_df,df_tmp])
else:
    key_df = pd.read_csv(args.training_key,sep='\t')
GSM_trn = np.array(key_df.iloc[:,0])
GSE_trn = np.array(key_df.iloc[:,1])
# get measured genes
data_GeneIDs = df_data['GeneIDs'].to_numpy()
data_GeneIDs = np.array([str(item) for item in data_GeneIDs])

if args.targets != 'None':
    targets = np.loadtxt(args.targets,dtype=str)
else:
    targets = np.setdiff1d(GeneIDs,data_GeneIDs)


print('Getting rid of target genes if they are in the measured gene list')
targets = np.setdiff1d(targets,data_GeneIDs)

print('Checking which measured genes and targets are in traning data')
# see which targets can be imputed
target_inds = []
target_names = []
bad_targets = []
for atarget in targets:
    try:
        ind_ = np.where(GeneIDs==atarget)[0][0]
        target_inds.append(ind_)
        target_names.append(atarget)
    except IndexError:
        bad_targets.append(atarget)
data_inds = []
data_names = []
df_inds = []
bad_data = []
# see which measured genes are part of training set
for idx, adata_gene in enumerate(data_GeneIDs):
    try:
        ind_ = np.where(GeneIDs==adata_gene)[0][0]
        data_inds.append(ind_)
        data_names.append(adata_gene)
        df_inds.append(idx)
    except IndexError:
        bad_data.append(adata_gene)
print('The number of targets in the training set is',len(target_inds))
print('The number of targets not in the training set is',len(bad_targets))
print('The number of measured genes in the traning set is',len(data_inds))
print('The number of measured genes not in the traning set is',len(bad_data),'\n')

# load all the data
print('Loading the traning data')
if args.use_all_paper_data in ['Microarray','RNAseq']:
    datatype = args.use_all_paper_data
    TrnData = np.load('../data/%s_Trn_Exp.npy'%datatype)
    ValData = np.load('../data/%s_Val_Exp.npy'%datatype)
    TstData = np.load('../data/%s_Tst_Exp.npy'%datatype)
    FullData = np.concatenate((TrnData,ValData,TstData),axis=0)
    del TrnData; del ValData; del TstData
    FullData = np.transpose(FullData)
else:
    FullData = np.load(args.training_data)
    FullData = np.transpose(FullData)
print('The number of samples in the training set is ',FullData.shape[1])
Xtrn = FullData[data_inds,:]
Xtst = FullData[target_inds,:]
del FullData
# standarize the data
print('Standardizing the training data','\n')
std_scale = StandardScaler().fit(Xtrn)
Xtrn = std_scale.transform(Xtrn)
Xtst = std_scale.transform(Xtst)

pred_df = pd.DataFrame(target_names,columns=['GeneIDs'])
beta_df = pd.DataFrame()
for aGSM in list(df_data)[1:]:
    print('Starting imputation for',aGSM)
    ytrn = df_data[aGSM].to_numpy()
    ytrn = ytrn[df_inds]
    # This hyperparameter seems to be near the optimal value for most situations
    reg = Lasso(alpha=0.01,fit_intercept=True,normalize=False,precompute=False,
                 copy_X=True,max_iter=1000,tol=0.001,warm_start=False,positive=False,
                 random_state=None,selection='random')
    tic1 = time.time()
    reg.fit(Xtrn,ytrn)
    # get preds
    ypreds = reg.predict(Xtst)
    pred_df[aGSM] = ypreds
    # get beta info
    n = 100 # this sets how many of top betas to get
    betas = reg.coef_
    beta_args = np.flip(np.argsort(betas))
    beta_sort = betas[beta_args]
    top_betas = beta_sort[0:n]
    beta_df[aGSM + '_Beta'] = top_betas
    top_GSMs = GSM_trn[beta_args][0:n]
    beta_df[aGSM + '_GSM'] = top_GSMs
    top_GSEs = GSE_trn[beta_args][0:n]
    beta_df[aGSM + '_GSE'] = top_GSEs
    print(aGSM,'took',reg.n_iter_,'iterations to run')
    print(aGSM,'took',time.time()-tic1,'seconds to train/predict','\n')
    
# save the files
save_dir = '../user_results/%s/'%rightnow
print('Saving the files to',save_dir,'\n')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pred_df.to_csv(save_dir+'predictions.tsv',sep='\t',header=True,index=False,float_format='%.5f')
beta_df.to_csv(save_dir+'top_betas.tsv',sep='\t',header=True,index=False,float_format='%.6f')
np.savetxt(save_dir+'unusable_targets.txt',bad_targets,fmt='%s')
np.savetxt(save_dir+'unusable_measured_genes.txt',bad_data,fmt='%s')

print('This script took %i minutes to run '%((time.time()-tic0)/60))



