'''
Dated: 10-17-2017, Author: Chris Mancuso, contact at: mancus16@msu.edu
Features:
1. The models in this script will be an l-hidden layer DNN with the same number of hidden units in each layer
'''

############################################################################################################

# This section of code imports the modules that will be used
import numpy as np
import sys
from scipy import stats
import argparse
import os
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras import optimizers
from keras import initializers
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae # tested with verison 0.20.3
from sklearn.metrics import mean_squared_error as mse # tested with verison 0.20.3
import pandas as pd # tested with verison 0.24.2
import time 
print('Modules loaded successfully\n')
############################################################################################################

tic0 = time.time()

# get the GPL to try from command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-g','--aGPL',
                    default = 'GPL571',
                    type = str,
                    help = 'The GPL to do')
args = parser.parse_args()
aGPL = args.aGPL

############################################################################################################  

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
Xtrn = Xdata_GPL570
ytrn = ydata_GPL570
Xtst = Xdata_aGPL
ytst = ydata_aGPL
std_scale = StandardScaler().fit(Xtrn)
Xtrn = std_scale.transform(Xtrn)
Xtst = std_scale.transform(Xtst)

############################################################################################################


# # The directory to the file used to store the csv logger
CSVlogger_filename   = fp_save  + '%s_DNN_CSVlogger.csv'%aGPL
# # The directory to the file used to check point the model
ModelCheck_filename   = fp_save + '%s_DNN_ModelCheck.hdf5'%aGPL
# The directory to the file used to store the evaluations of the test set
Evaluations_filename = fp_save + '%s_DNN_evals.tsv'%aGPL

############################################################################################################

# build the model 
arch = '9000,0.1,9000,0.1'
list1 = arch.split(',')
units = list1[::2]; units = [int(item) for item in units]
drps  = list1[1::2]; drps = [float(item) for item in drps] 
if len(units) != len(drps):
    print('The each hidden layer needs a dropout parameter')
        
# Set the architecture of the model
model = Sequential()
for idx in range(len(units)):
    if idx == 0:
        model.add(Dense(units=units[idx], input_dim=Xtrn.shape[1], activation='tanh',kernel_initializer='glorot_uniform'))
        model.add(Dropout(drps[idx]))
    else:
        model.add(Dense(units=units[idx], activation='tanh',kernel_initializer='glorot_uniform'))
        model.add(Dropout(drps[idx]))
model.add(Dense(units=ytrn.shape[1], activation='linear',kernel_initializer=initializers.RandomUniform(-1e-4,1e-4)))

# Set up the optimizer
myoptimize = optimizers.Adam(lr=1e-5)

model.compile(loss= 'mean_squared_error',
              optimizer=myoptimize,
              metrics=['mae']) 
# Set up Keras callback to log results in a csv file
csv_logger = CSVLogger(CSVlogger_filename)
model_checkpoint = ModelCheckpoint(ModelCheck_filename ,save_best_only=True ,save_weights_only=False ,monitor='val_loss')             
# Train the model.
History = model.fit(Xtrn, ytrn, 
                    epochs=200,
                    verbose=0,
                    batch_size=200, 
                    validation_data=(Xtst,ytst),
                    callbacks=[csv_logger,model_checkpoint],
                    shuffle=True)

############################################################################################################

# reload best weights (this should load the model with the lowest validation loss)
model.load_weights(ModelCheck_filename)

############################################################################################################
# Get the predictions for every single sample in the test set
# size of this should be samples x Nmodles

# get the predictions
preds = model.predict(Xtst, batch_size = 200)

print('Make and save evaluations')
# make the coloumn names and initial df
col_names = ['GPL','Model','Metric','Value','GeneIdx']
df_eval = pd.DataFrame(columns=col_names)    
# get metrics
mae_values    = mae(ydata_aGPL,preds,multioutput='raw_values')
df_eval = add_eval_to_df(df_eval, mae_values,'mae','DNN',col_names,aGPL)
rmse_values   = np.sqrt(mse(ydata_aGPL,preds,multioutput='raw_values')) # correct to have ytst_GL
cvrmse_values = rmse_values/np.mean(ydata_aGPL,axis=0) # correct to have ytst_GL
df_eval = add_eval_to_df(df_eval, cvrmse_values,'cvrmse','DNN',col_names,aGPL)
# save the dataframe
df_eval.to_csv(Evaluations_filename,sep='\t',header=True,index=False)


############################################################################################################        
print('\n\nThe script took %i minutes to run'%((time.time()-tic0)/60))

############################################################################################################
