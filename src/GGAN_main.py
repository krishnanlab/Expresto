import keras
import keras.layers as layers
import tensorflow as tf
from keras.models import Model
import numpy as np
import pandas as pd
import time
import argparse

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adadelta
from keras.callbacks import CSVLogger, ModelCheckpoint
from weightnorm import AdamWithWeightnorm
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
from scipy.spatial.distance import cosine # tested with verison 1.3.0
from scipy import stats


class GAN():
    def __init__(self,myopt,mylr,NumXgenes,NumYgenes,ArchType):

        if myopt == 'adam':
            optimizer = AdamWithWeightnorm(lr=mylr, beta_1=0.9,
                         beta_2=0.999,epsilon=1e-8)
        elif myopt == 'adadelta':
            optimizer= Adadelta()
        else:
            print('Not a valid optimizer')

        self.ld_genes_shape = NumXgenes
        self.target_genes_shape = NumYgenes
        self.ArchType = ArchType
        ###############################################
        # hard coded hyperparameters below
        self.gen_hidden_units = 9000
        self.disc_hidden_units = 3000

        self.leak_value = 0.2
        ## lower bernoulli p means higher number of zeros
        self.init_bernoulli_p = 0.1 ## Need to change with each iteration or epoch
        self.final_bernoulli_p = 0.99
        self.u1 = 0.1
        self.u2 = 0.15

        ## lambdas
        self.lbda_adv = 0.1
        self.lbda_cons = 1
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.custom_discriminator_loss(),
                                   optimizer=optimizer,
                                   metrics=['mae'])

        self.generator = self.build_generator(drop_frac = 0)
        self.generator_u1 = self.build_generator(drop_frac = self.u1)
        self.generator_u2 = self.build_generator(drop_frac = self.u2)
        self.generator_u1.trainable = False
        self.generator_u2.trainable = False



        z = Input(shape=(self.ld_genes_shape,))
        distrib = Input(shape=(self.target_genes_shape,))
        target_predicted_u1 = Input(shape=(1,))

        target_predicted = self.generator(z)

        self.discriminator.trainable = False
        lambda_mult = layers.Lambda(lambda x: tf.multiply(x[0],x[1]))
        target_predicted_masked = lambda_mult([distrib,target_predicted])
        
        validity_predicted = self.discriminator([z,target_predicted_masked])
        concat_out = layers.concatenate([target_predicted,validity_predicted],axis =1)

        self.combined = Model(inputs=[z,distrib,target_predicted_u1], 
                        outputs=concat_out)
        self.combined.compile(loss=self.custom_generator_loss(), 
              optimizer=optimizer)
                 


    def custom_discriminator_loss(self):
        def total_loss(y_true,y_pred):  
            validity_actual = y_true
            validity_predicted = y_pred
            ones_tensor = tf.ones(tf.shape(validity_actual))
            temp_tensor_ladv = tf.math.add(tf.math.square(tf.subtract(validity_actual,ones_tensor)),
                                tf.math.square(validity_predicted))
            L_adv = tf.reduce_sum(temp_tensor_ladv,0)/2
            return L_adv
        return total_loss

    def custom_generator_loss(self):
        def total_loss(y_true,y_pred):
            validity_actual = y_true[:,self.target_genes_shape:]
            validity_predicted = y_pred[:,self.target_genes_shape:]
            ones_tensor = tf.ones(tf.shape(validity_actual))
            L_adv = tf.math.square(tf.subtract(validity_predicted,ones_tensor))

            
            target_actual = y_true[:,:self.target_genes_shape]
            target_predicted = y_pred[:,:self.target_genes_shape]
            L_1 = tf.norm(target_actual-target_predicted,
                ord=1,axis = 1)

            return tf.reduce_sum(L_1) + tf.math.reduce_mean(self.lbda_adv*L_adv) \
                 + self.lbda_cons*tf.math.reduce_mean(self.combined.input[2])
        return total_loss
            
    def simple_model(self):
        units = [200,200,200]
        drps = [0.1,0.1,0.1]
        
        model = Sequential()
        for idx in range(len(units)):
            if idx == 0:
                model.add(Dense(units=units[idx], 
                    input_dim=self.ld_genes_shape, activation='tanh',kernel_initializer='glorot_uniform'))
                model.add(Dropout(drps[idx]))
            else:
                model.add(Dense(units=units[idx], activation='tanh',kernel_initializer='glorot_uniform'))
                model.add(Dropout(drps[idx]))
        model.add(Dense(units=self.target_genes_shape, 
                    activation='linear',kernel_initializer=keras.initializers.RandomUniform(-1e-4,1e-4)))
        return model



    def build_generator(self, drop_frac= 0):
        
        inp1 = Input(shape=(self.ld_genes_shape,))

        inp1_dp = Dropout(drop_frac)(inp1, training = True)

        h1 = Dense(self.gen_hidden_units)(inp1_dp)
        h1_dp = Dropout(drop_frac)(h1, training = True)
        h1_activated = LeakyReLU(alpha=self.leak_value)(h1_dp)

        if self.ArchType == '2L9000U':
            h2 = Dense(self.gen_hidden_units)(h1_activated)
            h2_add = Add()([h2,h1_activated])
            h2_add_dp = Dropout(drop_frac)(h2_add, training = True)
            # h2_add = Add()[h2_add,inp1]
            h2_activated = LeakyReLU(alpha=self.leak_value)(h2_add_dp)
            out = Dense(self.target_genes_shape)(h2_activated)
        elif self.ArchType == '3L9000U':
            h2 = Dense(self.gen_hidden_units)(h1_activated)
            h2_add = Add()([h2,h1_activated])
            h2_add_dp = Dropout(drop_frac)(h2_add, training = True)
            # h2_add = Add()[h2_add,inp1]
            h2_activated = LeakyReLU(alpha=self.leak_value)(h2_add_dp)
            h3 = Dense(self.gen_hidden_units)(h2_activated)
            h3_add = Add()([h3,h2_activated])
            h3_add = Add()([h3_add,h1_activated])
            h3_add_dp = Dropout(drop_frac)(h3_add, training = True)
            # h3_add = Add()[h3_add,inp1]
            h3_activated = LeakyReLU(alpha=self.leak_value)(h3_add_dp)
            out = Dense(self.target_genes_shape)(h3_activated)
        else:
            print('Not a valid arch type')
        
        return Model(inp1, out)

    def build_discriminator(self):

        model = Sequential()
        
        model.add(Dense(units=self.disc_hidden_units,
     		input_dim=self.ld_genes_shape+self.target_genes_shape, 
     		kernel_initializer='glorot_uniform'))
        model.add(LeakyReLU(alpha=self.leak_value))
    
        model.add(Dense(units=1,
     			kernel_initializer='glorot_uniform'))
        model.add(LeakyReLU(alpha=self.leak_value))
        model.summary()
        ld_input = Input(shape=(self.ld_genes_shape,))
        target_input = Input(shape=(self.target_genes_shape,))
        inp_disc =layers.concatenate([ld_input,target_input],axis=-1)
        validity = model(inp_disc)
        return Model([ld_input,target_input], validity)

    def add_eval_to_df(self,df_eval,split,train_data,test_data,metric_values,metric_name,
                        start_model,stop_model,FN_end,col_names,make_list='yes'):
        gene_inds = list(range(start_model,stop_model))
        num_genes = len(gene_inds)
        fill_data = []
        fill_data.append([FN_end] * num_genes)
        fill_data.append(['NA'] * num_genes)
        fill_data.append([train_data] * num_genes)
        fill_data.append([test_data] * num_genes)
        fill_data.append([split] * num_genes)
        fill_data.append(['%s'%genesplit] * num_genes)
        fill_data.append([metric_name] * num_genes)
        if make_list == 'yes':
            metric_values = list(metric_values)
        fill_data.append(metric_values)
        fill_data.append(gene_inds)
        fill_zip = list(zip(*fill_data))
        df_tmp = pd.DataFrame(fill_zip,columns=col_names)
        df_eval = pd.concat([df_eval,df_tmp])
        return df_eval

    def add_metric_values(self,yval_preds, ytst_preds):
        # make the coloumn names and initial df
        col_names = ['Model','HyperParameter','Train-Data','Test-Data','Split','GeneSplit','Metric','Value','SampleNum']
        df_eval = pd.DataFrame(columns=col_names)
        
        # R2 eval
        r2_val   = r2_score(yval_slice,yval_preds,multioutput='raw_values')
        df_eval = self.add_eval_to_df(df_eval,'Val',train_data,test_data,
                                r2_val,'r2',start_model,stop_model,FN_end,col_names,make_list='yes')
        r2_tst  = r2_score(ytst_slice,ytst_preds,multioutput='raw_values')
        df_eval = self.add_eval_to_df(df_eval,'Tst',train_data,test_data,
                                 r2_tst,'r2',start_model,stop_model,FN_end,col_names,make_list='yes')
        # mae eval
        mae_val   = mean_absolute_error(yval_slice,yval_preds,multioutput='raw_values')
        df_eval = self.add_eval_to_df(df_eval,'Val',train_data,test_data,
                                 mae_val,'mae',start_model,stop_model,FN_end,col_names,make_list='yes')
        mae_tst  =  mean_absolute_error(ytst_slice,ytst_preds,multioutput='raw_values')
        df_eval = self.add_eval_to_df(df_eval,'Tst',train_data,test_data,
                                 mae_tst,'mae',start_model,stop_model,FN_end,col_names,make_list='yes')
        # mse eval 
        rmse_val   = np.sqrt(mean_squared_error(yval_slice,yval_preds,multioutput='raw_values'))
        df_eval = self.add_eval_to_df(df_eval,'Val',train_data,test_data,
                                 rmse_val,'rmse',start_model,stop_model,FN_end,col_names,make_list='yes')
        rmse_tst   = np.sqrt(mean_squared_error(ytst_slice,ytst_preds,multioutput='raw_values'))
        df_eval = self.add_eval_to_df(df_eval,'Tst',train_data,test_data,
                                 rmse_tst,'rmse',start_model,stop_model,FN_end,col_names,make_list='yes')
        # cvrms eval
        cvrmse_val = rmse_val/np.mean(yval_slice,axis=0)
        df_eval = self.add_eval_to_df(df_eval,'Val',train_data,test_data,
                                 cvrmse_val,'cvrmse',start_model,stop_model,FN_end,col_names,make_list='yes')
        cvrmse_tst = rmse_tst/np.mean(ytst_slice,axis=0)
        df_eval = self.add_eval_to_df(df_eval,'Tst',train_data,test_data,
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
        df_eval = self.add_eval_to_df(df_eval,'Val',train_data,test_data,
                                 rhos_val,'spearman',start_model,stop_model,FN_end,col_names,make_list='no')
        df_eval = self.add_eval_to_df(df_eval,'Tst',train_data,test_data,
                                 rhos_tst,'spearman',start_model,stop_model,FN_end,col_names,make_list='no')
        df_eval = self.add_eval_to_df(df_eval,'Val',train_data,test_data,
                                 rho2s_val,'pearson',start_model,stop_model,FN_end,col_names,make_list='no')
        df_eval = self.add_eval_to_df(df_eval,'Tst',train_data,test_data,
                                 rho2s_tst,'pearson',start_model,stop_model,FN_end,col_names,make_list='no')
        df_eval = self.add_eval_to_df(df_eval,'Val',train_data,test_data,
                                 cosines_val,'cosine',start_model,stop_model,FN_end,col_names,make_list='no')
        df_eval = self.add_eval_to_df(df_eval,'Tst',train_data,test_data,
                                 cosines_tst,'cosine',start_model,stop_model,FN_end,col_names,make_list='no')
        return df_eval


    def train(self, epochs, batch_size,
                X_train,Y_train,
                X_val,Y_val,
                X_test,Y_test):
        np.random.seed(seed=0)

        total_gps = X_train.shape[0]

        best_model = ['',''] # this first is for the best epoch and the second ins the results df
        best_mae = 100000
        train_loss = []
        for epoch in range(epochs):
            ## set before each epoch 
            self.generator_u1.set_weights(self.generator.get_weights())
            self.generator_u2.set_weights(self.generator.get_weights())
            all_IDs = np.arange(0,total_gps)

            np.random.shuffle(all_IDs)
            random_ids_to_put = total_gps%batch_size
            total_batches = total_gps//batch_size
            if random_ids_to_put>0:
                idx = np.random.randint(0, total_gps, random_ids_to_put)
                all_IDs = np.concatenate((all_IDs,idx))
                total_batches = total_gps//batch_size + 1
            self.bernoulli_p = self.init_bernoulli_p + \
                              (self.final_bernoulli_p-self.init_bernoulli_p)*(epoch/(epochs-1))
            complete_distrib = np.random.binomial(n=1, p=self.bernoulli_p,
                                        size=(total_gps,self.target_genes_shape))
            for batchNo in range(total_batches):
            
                ## gene profiles
                idx = all_IDs[batchNo*batch_size:(batchNo+1)*batch_size]
                z = X_train[idx]
                target_actual = Y_train[idx]
                distrib = complete_distrib[idx]

                
                target_actual_masked = np.multiply(distrib,target_actual)/self.bernoulli_p
                validity_actual = self.discriminator.predict([z,target_actual_masked]) ## not sure about this, confused
                
                target_predicted = self.generator.predict(z)
                target_predicted_masked = np.multiply(distrib,target_predicted)/self.bernoulli_p
                disc_loss= self.discriminator.train_on_batch([z,target_predicted_masked],validity_actual)
            
                ## for L_cons           
                target_predicted_u1 = self.generator_u1.predict(z)
                target_predicted_u2 = self.generator_u2.predict(z)
                subtracted_noise = np.linalg.norm(target_predicted_u1-target_predicted_u2,
                                     axis = 1)**2
                concatenated_actual = np.concatenate((target_actual,validity_actual),axis = 1)
                ## for generator loss
                generator_loss = self.combined.train_on_batch([z,distrib,subtracted_noise],
                                  concatenated_actual)
        
            print("{} [D loss: {}] [G loss: {}]" .format(epoch, disc_loss, generator_loss))
            val_predicted = self.generator.predict(X_val) 
            test_predicted = self.generator.predict(X_test)
            mae_val_loss = mean_absolute_error(Y_val,val_predicted)
            mae_test_loss = mean_absolute_error(Y_test,test_predicted)
            train_loss.append([epoch,mae_val_loss,mae_test_loss])
            print("at {}, MAE loss: {}" .format(epoch, mae_val_loss))
            if mae_val_loss < best_mae:
                df_eval = self.add_metric_values(val_predicted,test_predicted)
                best_model = [epoch,df_eval]
                df_train_loss = pd.DataFrame(train_loss,columns=['epoch','val_mae','test_mae'])
                best_model[1].to_csv(Evaluations_filename,sep='\t',header=True,index=False)
                df_train_loss.to_csv(CSVlogger_filename,header=True,index=False)
                best_mae = mae_val_loss
        return best_model, train_loss


if __name__ == '__main__':
    
    ############################################################################################################
    tic = time.time()
    ############################################################################################################

    # This section adds the arguments using argparse (this allows args to be added in the command line)
    parser = argparse.ArgumentParser()

    parser.add_argument('-stm','--start_model',default = 0, type = int,
        help = 'The index of the target gene array to start with')
    parser.add_argument('-spm','--stop_model',default = 20, type = int,
        help = 'The index of the target gene array to stop with')
    parser.add_argument('-lr', default = 1e-6, type = float,
        help = 'This sets the learning rate')
    parser.add_argument('-opt','--optimize_type', default = 'adam',
        help = 'This sets the type of optimizer (adam,adadelta)')
    parser.add_argument('-trn','--train_data', default = 'Microarray', type = str,
        help = 'Training data to use')
    parser.add_argument('-tst','--test_data', default = 'Microarray', type = str,
        help = 'Test data to use (this will be used for val data too)')
    parser.add_argument('-a','--arch', default = '2L9000U', type = str,
        help = 'model arch L layer, U units , DN is DenseNet')
    parser.add_argument('-gs','--genesplit',default = 'LINCS',type = str,
        help = 'GPL96-570, LINCS')
    args = parser.parse_args()
    start_model       = args.start_model
    stop_model        = args.stop_model
    lr                = args.lr
    optimize_type     = args.optimize_type
    arch              = args.arch
    train_data        = args.train_data
    test_data         = args.test_data
    genesplit         = args.genesplit
    #####################################
    
    ############################################################################################################  

    save_dir = '../reproduce_results/GGAN-save/'%(train_data,test_data,genesplit)
    if optimize_type == 'adadelta':
        FN_end = 'opt--%s__lr--NA__mom--NA__arch--%s__start--%s__stop--%s'%(optimize_type,
                                                                           arch,start_model,stop_model)
    elif optimize_type == 'adam':
        FN_end = 'opt--%s__lr--%s__mom--NA__arch--%s__start--%s__stop--%s'%(optimize_type,lr,
                                                                           arch,start_model,stop_model)
    else:
        print('Not a valid optimizer')

    ########## load the data and inds to slice data #####################
    print('Loading the data')
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
    Xgenes = np.loadtxt(data_dir + '%s_Xgenes_inds.txt'%genesplit, dtype=int)
    ygenes = np.loadtxt(data_dir + '%s_ygenes_inds.txt'%genesplit, dtype=int) 
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
    print('Setting up some output file names')
    # # The directory to the file used to store the csv logger
    CSVlogger_filename   = save_dir + FN_end +'__CSVlogger.csv'
    # # The directory to the file used to store the evaluations of the test set
    Evaluations_filename = save_dir + FN_end +'__evals.tsv'
    
    
    ############################################################################################################
    print('Inializating GGAN')
    gan = GAN(optimize_type, lr, Xtrn.shape[1], ytrn_slice.shape[1], arch)
    print('Starting to Train')
    best_model, train_loss = gan.train(epochs=200, batch_size=200, 
              X_train=Xtrn,
              Y_train=ytrn_slice,
              X_val = Xval,
              Y_val = yval_slice,
              X_test = Xtst,
              Y_test = ytst_slice)
    print('The best epoch was', best_model[0])