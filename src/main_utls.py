import numpy as np # tested with verison 1.16.4
from sklearn.preprocessing import StandardScaler # tested with verison 0.20.3
from sklearn.neighbors import KNeighborsRegressor # tested with verison 0.20.3
from sklearn.linear_model import Lasso # tested with verison 0.20.3


def load_data(TrnData,TestData,Split,GeneSplit):
    
    data_dir = '../data/'
    
    # load dict to store all the data
    data_dict = {}
    

    Xgene_inds = np.loadtxt(data_dir+'%s_Xgenes_inds.txt'%GeneSplit,dtype=int)
    ygene_inds = np.loadtxt(data_dir+'%s_ygenes_inds.txt'%GeneSplit,dtype=int)
    
    # load the traning data
    trndata = np.load(data_dir+'%s_Trn_Exp.npy'%TrnData)
    if TrnData == 'RNAseq':
        trndata = np.arcsinh(trndata)
    Xtrn = trndata[:,Xgene_inds]
    ytrn = trndata[:,ygene_inds]
    data_dict['Xtrn-'+TrnData] = Xtrn
    data_dict['ytrn-'+TrnData] = ytrn
    
    # load the data to be used a test data
    for datatype in TestData:
            tstdata = np.load(data_dir+'%s_%s_Exp.npy'%(datatype,Split))
            if Split == 'Val':
                trimmed_inds = np.load(data_dir+'%s_trimmed_Val_inds.npy'%datatype)
                tstdata = tstdata[trimmed_inds,:]
            if datatype == 'RNAseq':
                tstdata = np.arcsinh(tstdata)
            Xtst = tstdata[:,Xgene_inds]
            ytst = tstdata[:,ygene_inds]
            data_dict['Xtst-'+datatype] = Xtst
            data_dict['ytst-'+datatype] = ytst
    return data_dict


def reorder_and_transpose(data_dict,TrnData,TestData):
    
    if TrnData == TestData[0]:
        data_dict['ytrn-'+TrnData], data_dict['Xtst-'+TrnData] = data_dict['Xtst-'+TrnData], data_dict['ytrn-'+TrnData]
    else:
        data_dict['ytrn-'+TestData[0]] = data_dict['Xtst-'+TestData[0]]
        data_dict['Xtst-'+TrnData] = data_dict['ytrn-'+TrnData]
        data_dict.pop('Xtst-'+TestData[0], None)
        data_dict.pop('ytrn-'+TrnData, None)
        
    for akey in data_dict:
        data_dict[akey] = np.transpose(data_dict[akey])
    
    return data_dict
    
def stdscale_data(data_dict,TrnData,TestData,Model):

    std_scale = StandardScaler().fit(data_dict['Xtrn-'+TrnData])
    # transform the Xtrn data
    data_dict['Xtrn-'+TrnData] = std_scale.transform(data_dict['Xtrn-'+TrnData])
    
    # use Xtrn scale to transform Val data
    if Model in ['SampleLasso','GeneKNN']:
        data_dict['Xtst-'+TrnData] = std_scale.transform(data_dict['Xtst-'+TrnData])
    else:
        for datatype in TestData:
            data_dict['Xtst-'+datatype] = std_scale.transform(data_dict['Xtst-'+datatype])
            
    return data_dict
    
def initialize_reg(Model,HyperParameter):
    
    if Model in ['SampleLasso','GeneLasso']:
        reg = Lasso(alpha=HyperParameter,fit_intercept=True,normalize=False,precompute=False,
                     copy_X=True,max_iter=1000,tol=0.001,warm_start=False,positive=False,
                     random_state=None,selection='random')
    elif Model in ['SampleKNN', 'GeneKNN']:
        reg = KNeighborsRegressor(n_neighbors=int(HyperParameter),weights='distance',algorithm='brute',
                                  leaf_size=30,p=2,metric='minkowski',metric_params=None,n_jobs=None)
    return reg