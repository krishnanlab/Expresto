import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import time

'''
1. I'm changing where the data is and making it as simple as possible
2. I'm trying to have it call no extra scripts

'''

############################################################################################################
class QN():
    def __init__(self):
        pass
    
    def fit(self,ref_data):
        '''
        This function will take in a matrix of data and
        output a reference distribution from it
        '''
        # sort the data
        sorted_data = np.sort(ref_data, axis=1)
        # find the column means to get the reference distribution
        # the self makes it so other parts of the class can see it
        self.ref_distrib = np.mean(sorted_data, axis = 0)
        # return the instance of the class with the updated fit
        return self
        
    def transform(self,target_data):
        order = target_data.argsort(axis=1)
        ranks = order.argsort(axis=1)
        transformed_data = self.ref_distrib[ranks]
        return transformed_data

def add_eval_to_df(df_eval,metric_values,metric_name,norm_name,col_names):
    num_genes = len(metric_values)
    gene_inds = list(np.arange(num_genes))
    fill_data = []
    fill_data.append([norm_name] * num_genes)
    fill_data.append([metric_name] * num_genes)
    metric_values = list(metric_values)
    fill_data.append(metric_values)
    fill_data.append(gene_inds)
    fill_zip = list(zip(*fill_data))
    df_tmp = pd.DataFrame(fill_zip,columns=col_names)
    df_eval = pd.concat([df_eval,df_tmp])
    return df_eval

############################################################################################################
# Hard code some of the paths
data_fp = '../data/'
fp_save = '../reproduce_results/Normalization-save/'
############################################################################################################
time0 = time.time()
#This section will split the data up in train, val, and test sets
print('Loading the Data')
train = np.load(data_fp + 'RNAseq_Trn_Exp.npy')
test  = np.load(data_fp + 'Microarray_Tst_Exp.npy')
print('Loading the data took',int((time.time()-time0)/60),'minutes to run','\n')

# QN transform the data
time1 = time.time()
print('Doing QN normalization')
myQN = QN().fit(test)
train_QN = myQN.transform(train)
test = myQN.transform(test)
print('QN took',int((time.time()-time1)/60),'minutes to run','\n')

# Load Gene indices for genes in  both GPL and Archs4 data for X and splits based on GPL splits between platforms
X_genes = np.loadtxt(data_fp + 'LINCS_Xgenes_inds.txt',dtype=int)
y_genes = np.loadtxt(data_fp + 'LINCS_ygenes_inds.txt',dtype=int)

time2 = time.time()      
print('Slicing and Standarizing the Data')
X_train  = np.transpose(train[:,X_genes])
y_train  = np.transpose(train[:,y_genes])
X_train_QN  = np.transpose(train_QN[:,X_genes])
y_train_QN  = np.transpose(train_QN[:,y_genes])
# only select 100 test samples to impute
np.random.seed(8)
test_samples = np.random.choice(np.arange(test.shape[0]), size=1000, replace=False, p=None)
test_slice = test[test_samples,:]
X_test  = np.transpose(test_slice[:,X_genes])
y_test  = test_slice[:,y_genes]

# standarize the data
std_scale = StandardScaler().fit(X_train)
X_train   = std_scale.transform(X_train)
y_train   = std_scale.transform(y_train)
# standarize the data the QN data
std_scale_QN = StandardScaler().fit(X_train_QN)
X_train_QN   = std_scale.transform(X_train_QN)
y_train_QN   = std_scale.transform(y_train_QN)
print('Slicing and Standarizing took',int((time.time()-time2)/60),'minutes to run','\n')

time3 = time.time()  
print('Doing SampleLasso')
preds = np.zeros((y_test.shape[0],y_test.shape[1]),dtype=float)
preds_QN = np.zeros((y_test.shape[0],y_test.shape[1]),dtype=float)
for idx in range(X_test.shape[1]):
    # do for untransformed train data
    time3a = time.time()
    clf = Lasso(alpha=0.1,fit_intercept=True,normalize=False,precompute=False,copy_X=True,max_iter=1000,tol=0.001,
                warm_start=False,positive=False,random_state=None,selection='random')
    clf.fit(X_train,X_test[:,idx])
    y_preds = clf.predict(y_train)
    preds[idx,:] = y_preds
    print('Doing Log Reg No Normalization on sample',idx,'took',int((time.time()-time3a)/60),'minutes to run')
    # do for  QN transformed train data
    time3b = time.time()
    clf = Lasso(alpha=0.1,fit_intercept=True,normalize=False,precompute=False,copy_X=True,max_iter=1000,tol=0.001,
                warm_start=False,positive=False,random_state=None,selection='random')
    clf.fit(X_train_QN,X_test[:,idx])
    y_preds = clf.predict(y_train_QN)
    preds_QN[idx,:] = y_preds
    print('Doing Log Reg QN on sample',idx,'took',int((time.time()-time3b)/60),'minutes to run')
print('Doing Log Reg total took',int((time.time()-time3)/60),'minutes to run','\n')

time4 = time.time()
print('Evaluating pedictions and saving the results')
# make the coloumn names and initial df
col_names = ['Normalization','Metric','Value','SampleNum']
df_eval = pd.DataFrame(columns=col_names)    
# get unnormalize train metrics
mae_values    = mae(y_test,preds,multioutput='raw_values')
df_eval = add_eval_to_df(df_eval, mae_values,'mae','No Normalization',col_names)
rmse_values   = np.sqrt(mse(y_test,preds,multioutput='raw_values'))
cvrmse_values = rmse_values/np.mean(test[:,y_genes],axis=0) # use full test to get avg expression for the gene to compare to main paper way
df_eval = add_eval_to_df(df_eval, cvrmse_values,'cvrmse','No Normalization',col_names)
# get  QN normalize train metrics  
mae_values_QN    = mae(y_test,preds_QN,multioutput='raw_values')
df_eval = add_eval_to_df(df_eval, mae_values_QN,'mae','QN',col_names)
rmse_values_QN   = np.sqrt(mse(y_test,preds_QN,multioutput='raw_values'))
cvrmse_values_QN = rmse_values_QN/np.mean(test[:,y_genes],axis=0) # use full test to get avg expression for the gene to compare to main paper way
df_eval = add_eval_to_df(df_eval, cvrmse_values_QN,'cvrmse','QN',col_names)
df_eval.to_csv(fp_save + 'Normalization_Comparison_SampleLasso.tsv',sep='\t')
print('Evaluating and Saving took',int((time.time()-time4)/60),'minutes to run','\n')

print('The whole script took',int((time.time()-time0)/60),'minutes to run')