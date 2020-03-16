import argparse # tested with python verison 3.7
import glob # tested with python verison 3.7
import numpy as np # tested with verison 1.16.4
from sklearn.metrics import r2_score # tested with verison 0.20.3
from sklearn.metrics import mean_absolute_error # tested with verison 0.20.3
from sklearn.metrics import mean_squared_error # tested with verison 0.20.3
from scipy import stats # tested with verison 1.3.0
from scipy.spatial.distance import cosine # tested with verison 1.3.0
import pandas as pd # tested with verison 0.24.2


def make_name_dict(afolder):
    name_dict = {}
    afolder_end = afolder.strip().split('/')[-1]
    folder_split = afolder_end.split('__')
    for item in folder_split:
        akey, avalue = item.split('--')
        name_dict[akey] = avalue
    return name_dict
    

def add_eval_to_df(df_eval,metric_values,metric_name,name_dict,ytst,col_names,make_list='yes'):
    mykeys = ['M','H','Trn','Tst','S','GS']
    num_genes = ytst.shape[1]
    fill_data = []
    for akey in mykeys:
        fill_data.append([name_dict[akey]] * num_genes)
    fill_data.append([metric_name] * num_genes)
    if make_list == 'yes':
        metric_values = list(metric_values)
    fill_data.append(metric_values)
    fill_data.append(list(range(num_genes)))
    fill_zip = list(zip(*fill_data))
    df_tmp = pd.DataFrame(fill_zip,columns=col_names)
    df_eval = pd.concat([df_eval,df_tmp])
    return df_eval


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('-sd','--savedir',
                        default = '../reproduce_results/LASSO-KNN-save/',
                        type = str,
                        help = 'The base dir where all the specific directories are')
    args = parser.parse_args()
    SaveBaseDir = args.savedir


    Folders = glob.glob(SaveBaseDir + '*')
    data_dict = {} # store multiple dataset so don't have to load every folder
    for afolder in Folders:
        # make a name_dict for the folder
        name_dict = make_name_dict(afolder)
        
        # get all npy files in the folder
        all_files = glob.glob(afolder+'/*')
        # check if evaluations file already exsists
        if afolder + '/evals.tsv' in all_files:
            continue
        # check if there is a summarized prediction file in the folder
        if afolder + '/preds.npy' not in all_files:
            continue
        else:
            preds = np.load(afolder + '/preds.npy')
            if name_dict['M'] == 'GeneKNN':
                preds = np.transpose(preds)
        
        data_dir = '../data/'
        data_name = '%s--%s'%(name_dict['Tst'],name_dict['S'])
        if data_name not in data_dict:
            tstdata = np.load(data_dir+'%s_%s_Exp.npy'%(name_dict['Tst'],name_dict['S']))
            if name_dict['S'] == 'Val':
                trimmed_inds = np.load(data_dir+'%s_trimmed_Val_inds.npy'%name_dict['Tst'])
                tstdata = tstdata[trimmed_inds,:]
            if name_dict['Tst'] == 'RNAseq':
                tstdata = np.arcsinh(tstdata)
            data_dict[data_name] = tstdata
        else:
            tstdata = data_dict[data_name]
        # slice the test data for proper gene split
        ygene_inds = np.loadtxt(data_dir+'%s_ygenes_inds.txt'%name_dict['GS'],dtype=int)
        ytst = tstdata[:,ygene_inds]
        
        # start making evals and adding to df
        # make an empty df with the correct column headers
        col_names = ['Model','HyperParameter','Train-Data','Test-Data','Split','GeneSplit','Metric','Value','SampleNum']
        df_eval = pd.DataFrame(columns=col_names)
        # r2
        r2 = r2_score(ytst,preds,multioutput='raw_values')
        df_eval = add_eval_to_df(df_eval,r2,'r2',name_dict,ytst,col_names,make_list='yes')
        # mae
        mae = mean_absolute_error(ytst,preds,multioutput='raw_values')
        df_eval = add_eval_to_df(df_eval,mae,'mae',name_dict,ytst,col_names,make_list='yes')
        # rmse
        rmse = np.sqrt(mean_squared_error(ytst,preds,multioutput='raw_values'))
        df_eval = add_eval_to_df(df_eval,rmse,'rmse',name_dict,ytst,col_names,make_list='yes')
        # cvrmse
        cvrmse = rmse/np.mean(ytst,axis=0)
        df_eval = add_eval_to_df(df_eval,cvrmse,'cvrmse',name_dict,ytst,col_names,make_list='yes')
        # do spearman, pearson and cosine
        rhos = [] # for spearman
        rho2s = [] # for pearson
        cosines = [] # for cosine similaritywhat a
        for idx in range(ytst.shape[1]): 
            rho, p = stats.spearmanr(ytst[:,idx],preds[:,idx],axis=0)
            rhos.append(rho)
            rho2, p2 = stats.pearsonr(ytst[:,idx],preds[:,idx])
            rho2s.append(rho2)
            cosine_ = -1 * cosine(ytst[:,idx],preds[:,idx]) + 1 # calculates cosine_similarity
            cosines.append(cosine_)
        df_eval = add_eval_to_df(df_eval,rhos,'spearman',name_dict,ytst,col_names,make_list='no')
        df_eval = add_eval_to_df(df_eval,rho2s,'pearson',name_dict,ytst,col_names,make_list='no')
        df_eval = add_eval_to_df(df_eval,cosines,'cosine',name_dict,ytst,col_names,make_list='no')
        
        # save the df
        df_eval.to_csv(afolder + '/evals.tsv',sep='\t',header=True,index=False)
        print('Evaluations saved for',afolder)
        print()
        