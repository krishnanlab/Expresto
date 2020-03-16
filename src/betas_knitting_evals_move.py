import argparse # tested with python verison 3.7
import glob # tested with python verison 3.7
import numpy as np # tested with verison 1.16.4
import os # tested with python verison 3.7
from sklearn.metrics import r2_score # tested with verison 0.20.3
from sklearn.metrics import mean_absolute_error # tested with verison 0.20.3
from sklearn.metrics import mean_squared_error # tested with verison 0.20.3
from scipy import stats # tested with verison 1.3.0
from scipy.spatial.distance import cosine # tested with verison 1.3.0
import pandas as pd # tested with verison 0.24.2
from shutil import copyfile
import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--scratch_dir',
                        default = '../reproduce_results/Beta-save/',
                        type = str,
                        help = 'Where scratch dir is')
    args = parser.parse_args()
    scratch_dir = args.scratch_dir
    # This was used becuase data orginally had to be moved out of temporary scratch
    # So moving doesn't need to be done any more
    final_save_dir  = args.scratch_dir


def make_name_dict(afolder):
    name_dict = {}
    afolder_end = afolder.strip().split('/')[-1]
    folder_split = afolder_end.split('__')
    for item in folder_split:
        akey, avalue = item.split('--')
        name_dict[akey] = avalue
    return name_dict


def add_eval_to_df(df_eval,model,metric_values,metric_name,ytst,col_names,make_list='yes'):
    if model == 'SL':
        mykeys = ['SampleLasso','0.001','Microarray','Microarray','Beta','GPL96-570']
    elif model == 'KNN':
        mykeys = ['SampleKNN','20','Microarray','Microarray','Beta','GPL96-570']
    else:
        print('Not a good model')
    num_genes = ytst.shape[1]
    fill_data = []
    for akey in mykeys:
        fill_data.append([akey] * num_genes)
    fill_data.append([metric_name] * num_genes)
    if make_list == 'yes':
        metric_values = list(metric_values)
    fill_data.append(metric_values)
    fill_data.append(list(range(num_genes)))
    fill_zip = list(zip(*fill_data))
    df_tmp = pd.DataFrame(fill_zip,columns=col_names)
    df_eval = pd.concat([df_eval,df_tmp])
    return df_eval



data_dir = '../data/' 

# load some files to get how big arrays should be
num_rows = len(np.loadtxt(data_dir + 'Beta_Tst_Inds.txt', dtype=int))
num_col_preds = len(np.loadtxt(data_dir + 'GPL96-570_ygenes_inds.txt', dtype=int))
num_col_betas = len(np.loadtxt(data_dir + 'Beta_Trn_Inds.txt', dtype=int))

# get all pred files for a specific model
pred_files = glob.glob(scratch_dir+'/Lasso_preds__ModelInds*npy')

pred_inds = [int(item.split('/')[-1].split('ModelInds--')[-1].split('.npy')[0]) for item in pred_files]
pred_inds = np.sort(np.array(pred_inds))
# check if all the files are there
full_inds = np.arange(num_rows)
if len(np.setdiff1d(full_inds,pred_inds)) > 0:
    print('The number of pred inds that have yet to be done for are', len(np.setdiff1d(full_inds,pred_inds)))
    raise ValueError('Number of preds inds not full')
else:
    preds = np.zeros((num_rows,num_col_preds))
    for aind in pred_inds:
        preds_tmp = np.load(scratch_dir + 'Lasso_preds__ModelInds--%i.npy'%aind)
        preds[aind,:] = preds_tmp
    np.save(scratch_dir + 'Lasso_preds.npy',preds)
    print('Predictions saved')

# get all beta files for a specific model
beta_files = glob.glob(scratch_dir+'/Lasso_betas__ModelInds*npy')

beta_inds = [int(item.split('/')[-1].split('ModelInds--')[-1].split('.npy')[0]) for item in beta_files]
beta_inds = np.sort(np.array(beta_inds))
# check if all the files are there
full_inds = np.arange(num_rows)
if len(np.setdiff1d(full_inds,beta_inds)) > 0:
    print('The number of beta inds that have yet to be done for are', len(np.setdiff1d(full_inds,beta_inds)))
    raise ValueError('Number of betas inds not full')
else:
    betas = np.zeros((num_rows,num_col_betas))
    for aind in beta_inds:
        betas_tmp = np.load(scratch_dir + 'Lasso_betas__ModelInds--%i.npy'%aind)
        betas[aind,:] = betas_tmp
    np.save(scratch_dir + 'Lasso_betas.npy',betas)
    print('Betas saved')
    

# make evals for Lasso and KNN

# load KNN preds
KNN_preds = np.load(scratch_dir + 'KNN_preds.npy')

# load tst set data
data = np.load(data_dir + 'Microarray_Trn_Exp.npy')
ygenes = np.loadtxt(data_dir + 'GPL96-570_ygenes_inds.txt', dtype=int)
tst_inds = np.loadtxt(data_dir + 'Beta_Tst_Inds.txt', dtype=int)
# slice data
ytst = data[tst_inds,:]
ytst = ytst[:,ygenes]

col_names = ['Model','HyperParameter','Train-Data','Test-Data','Split','GeneSplit','Metric','Value','SampleNum']
df_eval = pd.DataFrame(columns=col_names)

for apred in [[preds,'SL'], [KNN_preds,'KNN']]:
    # r2
    r2 = r2_score(ytst,apred[0],multioutput='raw_values')
    df_eval = add_eval_to_df(df_eval,apred[1],r2,'r2',ytst,col_names,make_list='yes')
    # mae
    mae = mean_absolute_error(ytst,apred[0],multioutput='raw_values')
    df_eval = add_eval_to_df(df_eval,apred[1],mae,'mae',ytst,col_names,make_list='yes')
    # rmse
    rmse = np.sqrt(mean_squared_error(ytst,apred[0],multioutput='raw_values'))
    df_eval = add_eval_to_df(df_eval,apred[1],rmse,'rmse',ytst,col_names,make_list='yes')
    # cvrmse
    cvrmse = rmse/np.mean(ytst,axis=0)
    df_eval = add_eval_to_df(df_eval,apred[1],cvrmse,'cvrmse',ytst,col_names,make_list='yes')
    # do spearman, pearson and cosine
    rhos = [] # for spearman
    rho2s = [] # for pearson
    cosines = [] # for cosine similarity
    for idx in range(ytst.shape[1]): 
        rho, p = stats.spearmanr(ytst[:,idx],apred[0][:,idx],axis=0)
        rhos.append(rho)
        rho2, p2 = stats.pearsonr(ytst[:,idx],apred[0][:,idx])
        rho2s.append(rho2)
        cosine_ = -1 * cosine(ytst[:,idx],apred[0][:,idx]) + 1 # calculates cosine_similarity
        cosines.append(cosine_)
    df_eval = add_eval_to_df(df_eval,apred[1],rhos,'spearman',ytst,col_names,make_list='no')
    df_eval = add_eval_to_df(df_eval,apred[1],rho2s,'pearson',ytst,col_names,make_list='no')
    df_eval = add_eval_to_df(df_eval,apred[1],cosines,'cosine',ytst,col_names,make_list='no')

    # save the df
    df_eval.to_csv(scratch_dir + '/betas_evals.tsv',sep='\t',header=True,index=False)
print('Evaluations saved')
print()
    
print('Moving fils')
files_to_move = ['Lasso_preds.npy','betas_evals.tsv','KNN,dists.npy','KNN_inds.npy','KNN_preds.npy','Lasso_betas.npy']
for afile in files_to_move:
    print('Copying over',afile)
    copyfile(scratch_dir + '/%s'%afile, final_save_dir + '%s'%afile)
    
     



