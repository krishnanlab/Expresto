import numpy as np # tested with verison 1.16.4
import pandas as pd
import time # tested with python verison 3.7
import argparse
from sklearn.preprocessing import StandardScaler # tested with verison 0.20.3
from sklearn.linear_model import Lasso # tested with verison 0.20.3
from sklearn.metrics import mean_absolute_error as mae # tested with verison 0.20.3
from sklearn.metrics import mean_squared_error as mse # tested with verison 0.20.3

tic0 = time.time()

# get the GPL to try from command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-g','--aGPL',
                    default = 'GPL571',
                    type = str,
                    help = 'The GPL to do')
args = parser.parse_args()
aGPL = args.aGPL

# add all functions at the top
def add_eval_to_df(df_eval,metric_values,metric_name,model_name,col_names,GPLID):
    num_genes = len(metric_values)
    gene_inds = list(np.arange(num_genes))
    fill_data = []
    fill_data.append([GPLID] * num_genes)
    fill_data.append([model_name] * num_genes)
    fill_data.append([metric_name] * num_genes)
    metric_values = list(metric_values)
    fill_data.append(metric_values)
    fill_data.append(gene_inds)
    fill_zip = list(zip(*fill_data))
    df_tmp = pd.DataFrame(fill_zip,columns=col_names)
    df_eval = pd.concat([df_eval,df_tmp])
    return df_eval

# set some paths
fp_gpl570 = '../data/'
fp_seek =  fp_gpl570 + '/Multiple_Platforms/'
fp_save = '../reproduce_results/SEEK-save/'

print('Load and standardize the data')
# load the GPL data
Xdata_aGPL = np.load(fp_seek + '%s_Xdata.npy'%aGPL)
ydata_aGPL = np.load(fp_seek + '%s_ydata.npy'%aGPL)
# load GPL570 gene inds
GPL570_Xgene_inds = np.loadtxt(fp_seek + 'GPL570_%s_Xgenes_inds.txt'%aGPL,dtype=int)
GPL570_ygene_inds = np.loadtxt(fp_seek + 'GPL570_ygenes_inds.txt',dtype=int)
# load the GPL570data
GPL570data = np.load(fp_gpl570 + 'Microarray_Trn_Exp.npy')
# slice the GPL570 along gene axis
Xdata_GPL570 = GPL570data[:,GPL570_Xgene_inds]
ydata_GPL570 = GPL570data[:,GPL570_ygene_inds]
# subset to 90 random samples
sample_inds = np.loadtxt(fp_seek + '%s_90_Sample_Inds.txt'%aGPL,dtype=int)
Xdata_aGPL = Xdata_aGPL[sample_inds,:]
ydata_aGPL = ydata_aGPL[sample_inds,:]

# standarize and format
Xtrn = np.transpose(Xdata_GPL570)
ytrn = np.transpose(Xdata_aGPL)
Xtst = np.transpose(ydata_GPL570)
ytst = np.transpose(ydata_aGPL)
std_scale = StandardScaler().fit(Xtrn)
Xtrn = std_scale.transform(Xtrn)
Xtst = std_scale.transform(Xtst)

# do the machine learning for sample lasso
print('Doing ML for SampleLasso')
preds = np.zeros((ytst.shape[1],ytst.shape[0]),dtype=float)
for idx in range(ytrn.shape[1]):
    tic1 = time.time()
    clf = Lasso(alpha=0.01,fit_intercept=True,normalize=False,precompute=False,copy_X=True,max_iter=1000,tol=0.001,
                warm_start=False,positive=False,random_state=None,selection='random')
    clf.fit(Xtrn,ytrn[:,idx])
    y_preds = clf.predict(Xtst)
    preds[idx,:] = y_preds
    print('It took', int((time.time()-tic1)/60), 'minutes for model %i to run'%idx)

print('Make and save evaluations')
# make the coloumn names and initial df
col_names = ['GPL','Model','Metric','Value','GeneIdx']
df_eval = pd.DataFrame(columns=col_names)    
# get metrics
mae_values    = mae(ydata_aGPL,preds,multioutput='raw_values')
df_eval = add_eval_to_df(df_eval, mae_values,'mae','SampleLASSO',col_names,aGPL)
rmse_values   = np.sqrt(mse(ydata_aGPL,preds,multioutput='raw_values')) # correct to have ytst_GL
cvrmse_values = rmse_values/np.mean(ydata_aGPL,axis=0) # correct to have ytst_GL
df_eval = add_eval_to_df(df_eval, cvrmse_values,'cvrmse','SampleLASSO',col_names,aGPL)
# save the dataframe
df_eval.to_csv(fp_save + '%s_SampleLASSO_evals.tsv'%aGPL,sep='\t',header=True,index=False)

print('It took', int((time.time()-tic0)/60), 'minutes for the script to run')