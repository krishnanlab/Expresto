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
from sklearn.metrics import mean_absolute_error as mae # tested with verison 0.20.3
from sklearn.metrics import mean_squared_error as mse # tested with verison 0.20.3



class GAN():
    def __init__(self,NumXgenes,NumYgenes):
        

        optimizer = AdamWithWeightnorm(lr=1e-3, beta_1=0.9,
                    beta_2=0.999,epsilon=1e-8)

        self.ld_genes_shape = NumXgenes
        self.target_genes_shape = NumYgenes
        ###############################################
        # hard coded hyperparameters below
        self.gen_hidden_units = 9000
        self.disc_hidden_units = 3000

        self.leak_value = 0.2
        ## lower bernoulli p means higher number of zeros
        self.init_bernoulli_p = 0.1 
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

        h2 = Dense(self.gen_hidden_units)(h1_activated)
        h2_add = Add()([h2,h1_activated])
        h2_add_dp = Dropout(drop_frac)(h2_add, training = True)
        h2_activated = LeakyReLU(alpha=self.leak_value)(h2_add_dp)
        out = Dense(self.target_genes_shape)(h2_activated)
        
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

        
    def add_eval_to_df(self, df_eval,metric_values,metric_name,model_name,col_names,GPLID):
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

    def add_metric_values(self, preds):
        # make the coloumn names and initial df
        col_names = ['GPL','Model','Metric','Value','GeneIdx']
        df_eval = pd.DataFrame(columns=col_names)

        mae_values    = mae(ydata_aGPL,preds,multioutput='raw_values')
        df_eval = self.add_eval_to_df(df_eval, mae_values,'mae','GGAN',col_names,aGPL)
        rmse_values   = np.sqrt(mse(ydata_aGPL,preds,multioutput='raw_values')) # correct to have ytst_GL
        cvrmse_values = rmse_values/np.mean(ydata_aGPL,axis=0) # correct to have ytst_GL
        df_eval = self.add_eval_to_df(df_eval, cvrmse_values,'cvrmse','GGAN',col_names,aGPL)
        
        return df_eval


    def train(self, epochs, batch_size,
                X_train,Y_train,
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
            ## would be a 1-D vector, need to be multiplied, adversial loss 
            ## bernoulli_p need to change with each iteration or epoch
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
            train_predicted = self.generator.predict(X_train)
            test_predicted = self.generator.predict(X_test)
            mae_train_loss = mae(Y_train,train_predicted)
            mae_test_loss = mae(Y_test,test_predicted)
            train_loss.append([epoch,mae_train_loss,mae_test_loss])
            df_train_loss = pd.DataFrame(train_loss,columns=['epoch','train_mae','test_mae'])
            df_train_loss.to_csv(CSVlogger_filename,header=True,index=False)
            print("at {}, MAE loss: {}" .format(epoch, mae_test_loss))
            if mae_test_loss < best_mae:
                df_eval = self.add_metric_values(test_predicted)
                best_model = [epoch,df_eval]
                best_model[1].to_csv(Evaluations_filename,sep='\t',header=True,index=False)
                df_train_loss.to_csv(CSVlogger_filename,header=True,index=False)
                best_mae = mae_test_loss
        return best_model, train_loss


if __name__ == '__main__':
    
    ############################################################################################################
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

    # set some paths
    fp_gpl570 = '../data/'
    fp_seek =  fp_gpl570 + '/Multiple_Platforms/'
    fp_save = '../reproduce_results/SEEK-save/'
    
    
    ############################################################################################################  

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
    print('Setting up some output file names')
    # # The directory to the file used to store the csv logger
    CSVlogger_filename   = fp_save  + '%s_GGAN_CSVlogger.csv'%aGPL
    # The directory to the file used to store the evaluations of the test set
    Evaluations_filename = fp_save + '%s_GGAN_evals.tsv'%aGPL
    
    ############################################################################################################
    print('Inializating GGAN')
    gan = GAN(Xtrn.shape[1], ytrn.shape[1])
    print('Starting to Train')
    best_model, train_loss = gan.train(epochs=200, batch_size=200, 
              X_train=Xtrn,
              Y_train=ytrn,
              X_test = Xtst,
              Y_test = ytst)
    print('The best epoch was', best_model[0])
    print('\n\nThe script took %i minutes to run'%((time.time()-tic0)/60))
    
    