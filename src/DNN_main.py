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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
from scipy.spatial.distance import cosine # tested with verison 1.3.0
import pandas as pd # tested with verison 0.24.2
import time 
print('Modules loaded successfully\n')
############################################################################################################
tic = time.time()

############################################################################################################

# This section adds the arguments using argparse (this allows args to be added in the command line)
parser = argparse.ArgumentParser()

parser.add_argument('-stm','--start_model',default = 40, type = int,
    help = 'The index of the target gene array to start with')
parser.add_argument('-spm','--stop_model',default = 45, type = int,
    help = 'The index of the target gene array to stop with')
parser.add_argument('-lr', default = 1e-3, type = float, 
    help = 'This sets the learning rate')
parser.add_argument('-opt','--optimize_type', default = 'adadelta', 
    help = 'This sets the type of optimizer (sgd,adam,adadelta)')
parser.add_argument('-m','--momentum', default = 0.5, type = float, 
    help = 'This sets the momentum in SGD')
parser.add_argument('-arch', default = '9000,0.1,9000,0.1', type = str, 
    help = 'Architecture of the model given hiddenunit1,dropout1,hiddenunit2,drop2...')
parser.add_argument('-trn','--train_data', default = 'RNAseq', type = str, 
    help = 'Training data to use')
parser.add_argument('-tst','--test_data', default = 'Microarray', type = str, 
    help = 'Test data to use (this will be used for val data too)')
parser.add_argument('-s','--save_dir', default = '../reproduce_results/DNN-save/', type = str, 
    help = 'Where to save data')
args = parser.parse_args()
start_model       = args.start_model
stop_model        = args.stop_model
lr                = args.lr
optimize_type     = args.optimize_type
momentum          = args.momentum
arch              = args.arch
train_data        = args.train_data
test_data         = args.test_data
save_dir          = args.save_dir

############################################################################################################  

save_dir = save_dir + '/%s__%s/'%(train_data,test_data)
if optimize_type == 'adadelta':
    FN_end = 'opt--%s__lr--NA__mom--NA__arch--%s__start--%s__stop--%s'%(optimize_type,
                                                                       arch,start_model,stop_model)
elif optimize_type == 'adam':
    FN_end = 'opt--%s__lr--%s__mom--NA__arch--%s__start--%s__stop--%s'%(optimize_type,lr,
                                                                       arch,start_model,stop_model)
elif optimize_type == 'sgd':
    FN_end = 'opt--%s__lr--%s__mom--%s__arch--%s__start--%s__stop--%s'%(optimize_type,lr,momentum,
                                                                       arch,start_model,stop_model)
else:
    print('Not a valid optimizer')

########## load the data and inds to slice data #####################
print('Loading the data')
# need to change this to /.. later
data_dir = '../data/'
trn_data = np.load(data_dir + '%s_Trn_Exp.npy'%train_data)
val_data = np.load(data_dir + '%s_Val_Exp.npy'%test_data)
tst_data = np.load(data_dir + '%s_Tst_Exp.npy'%test_data)
if train_data == 'RNAseq':
    trn_data = np.arcsinh(trn_data)
if test_data == 'RNAseq':
    val_data = np.arcsinh(val_data)
    tst_data = np.arcsinh(tst_data)
# load the gene inds files
Xgenes = np.loadtxt(data_dir + 'LINCS_Xgenes_inds.txt', dtype=int)
ygenes = np.loadtxt(data_dir + 'LINCS_ygenes_inds.txt', dtype=int) 
print()
  
############################################################################################################


print('Slicing and Standarizing Data')


Xtrn = trn_data[:,Xgenes]
ytrn = trn_data[:,ygenes]
ytrn_slice = ytrn[:,start_model:stop_model]
Xval = val_data[:,Xgenes]
yval = val_data[:,ygenes]
yval_slice = yval[:,start_model:stop_model]
Xtst = tst_data[:,Xgenes]
ytst = tst_data[:,ygenes]
ytst_slice = ytst[:,start_model:stop_model]

std_scale = StandardScaler().fit(Xtrn)
Xtrn = std_scale.transform(Xtrn)
Xval = std_scale.transform(Xval)
Xtst = std_scale.transform(Xtst)

############################################################################################################


# # The directory to the file used to store the csv logger
CSVlogger_filename   = save_dir + FN_end +'__CSVlogger.csv'
# # The directory to the file used to check point the model
ModelCheck_filename   = save_dir + FN_end +'__ModelCheck.hdf5'
# The directory to the file used to store the evaluations of the test set
Evaluations_filename = save_dir + FN_end +'__evals.tsv'

############################################################################################################

# build the model 
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
model.add(Dense(units=ytrn_slice.shape[1], activation='linear',kernel_initializer=initializers.RandomUniform(-1e-4,1e-4)))

# Set up the optimizer
if optimize_type == 'sgd':
    myoptimize = optimizers.SGD(lr=lr, momentum=momentum)
if optimize_type == 'adam':
    myoptimize = optimizers.Adam(lr=lr)
if optimize_type == 'adadelta':
    myoptimize = optimizers.Adadelta()
# Compile the model by selecting the loss function and the optimizer
model.compile(loss= 'mean_squared_error',
              optimizer=myoptimize,
              metrics=['mae']) 
# Set up Keras callback to log results in a csv file
csv_logger = CSVLogger(CSVlogger_filename)
model_checkpoint = ModelCheckpoint(ModelCheck_filename ,save_best_only=True ,save_weights_only=False ,monitor='val_loss')             
# Train the model.
History = model.fit(Xtrn, ytrn_slice, 
                    epochs=200,
                    verbose=0,
                    batch_size=200, 
                    validation_data=(Xval,yval_slice),
                    callbacks=[csv_logger,model_checkpoint],
                    shuffle=True)

############################################################################################################

# reload best weights (this should load the model with the lowest validation loss)
model.load_weights(ModelCheck_filename)

############################################################################################################
# Get the predictions for every single sample in the test set
# size of this should be samples x Nmodles


def add_eval_to_df(df_eval,split,train_data,test_data,metric_values,metric_name,
                    start_model,stop_model,FN_end,col_names,make_list='yes'):
    gene_inds = list(range(start_model,stop_model))
    num_genes = len(gene_inds)
    fill_data = []
    fill_data.append([FN_end] * num_genes)
    fill_data.append(['NA'] * num_genes)
    fill_data.append([train_data] * num_genes)
    fill_data.append([test_data] * num_genes)
    fill_data.append([split] * num_genes)
    fill_data.append(['LINCS'] * num_genes)
    fill_data.append([metric_name] * num_genes)
    if make_list == 'yes':
        metric_values = list(metric_values)
    fill_data.append(metric_values)
    fill_data.append(gene_inds)
    fill_zip = list(zip(*fill_data))
    df_tmp = pd.DataFrame(fill_zip,columns=col_names)
    df_eval = pd.concat([df_eval,df_tmp])
    return df_eval

# get the predictions
yval_preds = model.predict(Xval, batch_size = 200)
ytst_preds = model.predict(Xtst, batch_size = 200)

# make the coloumn names and initial df
col_names = ['Model','HyperParameter','Train-Data','Test-Data','Split','GeneSplit','Metric','Value','SampleNum']
df_eval = pd.DataFrame(columns=col_names)

# R2 eval
r2_val   = r2_score(yval_slice,yval_preds,multioutput='raw_values')
df_eval = add_eval_to_df(df_eval,'Val',train_data,test_data,
                        r2_val,'r2',start_model,stop_model,FN_end,col_names,make_list='yes')
r2_tst  = r2_score(ytst_slice,ytst_preds,multioutput='raw_values')
df_eval = add_eval_to_df(df_eval,'Tst',train_data,test_data,
                         r2_tst,'r2',start_model,stop_model,FN_end,col_names,make_list='yes')
# mae eval
mae_val   = mean_absolute_error(yval_slice,yval_preds,multioutput='raw_values')
df_eval = add_eval_to_df(df_eval,'Val',train_data,test_data,
                         mae_val,'mae',start_model,stop_model,FN_end,col_names,make_list='yes')
mae_tst  =  mean_absolute_error(ytst_slice,ytst_preds,multioutput='raw_values')
df_eval = add_eval_to_df(df_eval,'Tst',train_data,test_data,
                         mae_tst,'mae',start_model,stop_model,FN_end,col_names,make_list='yes')
# mse eval 
rmse_val   = np.sqrt(mean_squared_error(yval_slice,yval_preds,multioutput='raw_values'))
df_eval = add_eval_to_df(df_eval,'Val',train_data,test_data,
                         rmse_val,'rmse',start_model,stop_model,FN_end,col_names,make_list='yes')
rmse_tst   = np.sqrt(mean_squared_error(ytst_slice,ytst_preds,multioutput='raw_values'))
df_eval = add_eval_to_df(df_eval,'Tst',train_data,test_data,
                         rmse_tst,'rmse',start_model,stop_model,FN_end,col_names,make_list='yes')
# cvrms eval
cvrmse_val = rmse_val/np.mean(yval_slice,axis=0)
df_eval = add_eval_to_df(df_eval,'Val',train_data,test_data,
                         cvrmse_val,'cvrmse',start_model,stop_model,FN_end,col_names,make_list='yes')
cvrmse_tst = rmse_tst/np.mean(ytst_slice,axis=0)
df_eval = add_eval_to_df(df_eval,'Tst',train_data,test_data,
                         cvrmse_tst,'cvrmse',start_model,stop_model,FN_end,col_names,make_list='yes')
# spearman, pearson, cosine eval
rhos_val = [] # for spearman
rho2s_val = [] # for pearson
cosines_val = [] # for cosine similarity
rhos_tst = [] # for spearman
rho2s_tst = [] # for pearson
cosines_tst = [] # for cosine similarity
for idx in range(ytst_slice.shape[1]): 
    rho_val, p_val = stats.spearmanr(yval_slice[:,idx],yval_preds[:,idx],axis=0)
    rhos_val.append(rho_val)
    rho_tst, p_tst = stats.spearmanr(ytst_slice[:,idx],ytst_preds[:,idx],axis=0)
    rhos_tst.append(rho_tst)
    
    rho2_val, p2_val = stats.pearsonr(yval_slice[:,idx],yval_preds[:,idx])
    rho2s_val.append(rho2_val)
    rho2_tst, p2_tst = stats.pearsonr(ytst_slice[:,idx],ytst_preds[:,idx])
    rho2s_tst.append(rho2_tst)
    
    cosine_val = -1 * cosine(yval_slice[:,idx],yval_preds[:,idx]) + 1 # calculates cosine_similarity
    cosines_val.append(cosine_val)
    cosine_tst = -1 * cosine(ytst_slice[:,idx],ytst_preds[:,idx]) + 1 # calculates cosine_similarity
    cosines_tst.append(cosine_tst)
df_eval = add_eval_to_df(df_eval,'Val',train_data,test_data,
                         rhos_val,'spearman',start_model,stop_model,FN_end,col_names,make_list='no')
df_eval = add_eval_to_df(df_eval,'Tst',train_data,test_data,
                         rhos_tst,'spearman',start_model,stop_model,FN_end,col_names,make_list='no')
df_eval = add_eval_to_df(df_eval,'Val',train_data,test_data,
                         rho2s_val,'pearson',start_model,stop_model,FN_end,col_names,make_list='no')
df_eval = add_eval_to_df(df_eval,'Tst',train_data,test_data,
                         rho2s_tst,'pearson',start_model,stop_model,FN_end,col_names,make_list='no')
df_eval = add_eval_to_df(df_eval,'Val',train_data,test_data,
                         cosines_val,'cosine',start_model,stop_model,FN_end,col_names,make_list='no')
df_eval = add_eval_to_df(df_eval,'Tst',train_data,test_data,
                         cosines_tst,'cosine',start_model,stop_model,FN_end,col_names,make_list='no')


# save the df
df_eval.to_csv(Evaluations_filename,sep='\t',header=True,index=False)


############################################################################################################
toc = time.time()           
print('\n\nThe script took %i minutes to run'%((toc-tic)/60))

############################################################################################################
