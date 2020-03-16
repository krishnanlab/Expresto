import argparse # tested with python verison 3.7
import glob # tested with python verison 3.7
import numpy as np # tested with verison 1.16.4
import os # tested with python verison 3.7


'''
This script will knit both genes and betas. If makes a new combine file
it will delete all the other files
'''

def make_name_dict(afolder):
    name_dict = {}
    afolder_end = afolder.strip().split('/')[-1]
    folder_split = afolder_end.split('__')
    for item in folder_split:
        akey, avalue = item.split('--')
        name_dict[akey] = avalue
    return name_dict


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('-sd','--savedir',
                        default = '../reproduce_results/LASSO-KNN-save/',
                        type = str,
                        help = 'The base dir where all the specific directories are')
    args = parser.parse_args()
    SaveBaseDir = args.savedir

    data_dir = '../data/'

    Folders = glob.glob(SaveBaseDir + '*')
    
    # knit the preds if they are all there (save as a samplexgene array)
    for afolder in Folders:
        # make a name_dict for the folder
        name_dict = make_name_dict(afolder)

        if name_dict['M'] not in ['SampleLasso','GeneLasso']:
            continue

        #### Knit together the preds files #######
        # get the size of the final pred array
        if name_dict['S'] == 'Tst':
            num_samples = np.load(data_dir+'%s_%s_Exp.npy'%(name_dict['Tst'],name_dict['S']),mmap_mode='r').shape[0]
        elif name_dict['S'] == 'Val':
            num_samples = len(np.load(data_dir+'%s_trimmed_Val_inds.npy'%name_dict['Tst']))
        else:
            print('Not a valid split')
            continue
        num_genes = len(np.loadtxt(data_dir+'%s_ygenes_inds.txt'%name_dict['GS'],dtype=int))
        # get all pred files for a specific model
        pred_files = glob.glob(afolder+'/preds__ModelInds*npy')
        if len(pred_files) == 0:
            continue
        else:
            pred_inds = [int(item.split('/')[-1].split('ModelInds--')[-1].split('.npy')[0]) for item in pred_files]
            pred_inds = np.sort(np.array(pred_inds))
        # check if all the files are there
        if name_dict['M'] == 'SampleLasso':
            full_inds = np.arange(num_samples)
        if name_dict['M'] == 'GeneLasso':
            full_inds = np.arange(num_genes)
        if len(np.setdiff1d(full_inds,pred_inds)) > 0:
            print('The number of pred inds that have yet to be done for %s are'%afolder, len(np.setdiff1d(full_inds,pred_inds)))
            continue
        else:
            preds = np.zeros((num_samples,num_genes))
            for aind in pred_inds:
                preds_tmp = np.load(afolder + '/preds__ModelInds--%i.npy'%aind)
                if name_dict['M'] == 'SampleLasso':
                    preds[aind,:] = preds_tmp
                if name_dict['M'] == 'GeneLasso':
                    preds[:,aind] = preds_tmp
            np.save(afolder + '/preds.npy',preds)
            print('Predictions saved for', afolder)
            for aFN in pred_files:
                os.remove(aFN)
            print('Preds deleted for', afolder)
            print()
        
    # loop through again looking for beta files
    for afolder in Folders:  
        # make a name_dict for the folder
        name_dict = make_name_dict(afolder)

        if name_dict['M'] not in ['SampleLasso','GeneLasso']:
            continue
          
        #### Knit together the beta files ####
        if name_dict['M'] == 'SampleLasso':
            num_columns = np.load(data_dir+'%s_Trn_Exp.npy'%(name_dict['Trn']),mmap_mode='r').shape[0]
            if name_dict['S'] == 'Tst':
                num_rows = np.load(data_dir+'%s_Tst_Exp.npy'%(name_dict['Tst']),mmap_mode='r').shape[0]
            if name_dict['S'] == 'Val':
                num_rows = len(np.load(data_dir+'%s_trimmed_Val_inds.npy'%name_dict['Tst']))
        elif name_dict['M'] == 'GeneLasso':
            num_columns = len(np.loadtxt(data_dir+'%s_Xgenes_inds.txt'%name_dict['GS'],dtype=int))
            num_rows = len(np.loadtxt(data_dir+'%s_ygenes_inds.txt'%name_dict['GS'],dtype=int))
        else:
            print('Not a valid method')
            continue
        # get all beta files for a specific model
        beta_files = glob.glob(afolder+'/betas__ModelInds*npy')
        if len(beta_files) == 0:
            continue
        else:
            beta_inds = [int(item.split('/')[-1].split('ModelInds--')[-1].split('.npy')[0]) for item in beta_files]
            beta_inds = np.sort(np.array(beta_inds))
            # check if all the files are there
            full_inds = np.arange(num_rows)
            if len(np.setdiff1d(full_inds,beta_inds)) > 0:
                print('The number of beta inds that have yet to be done for %s are'%afolder, len(np.setdiff1d(full_inds,beta_inds)))
                continue
            else:
                betas = np.zeros((num_rows,num_columns))
                for aind in beta_inds:
                    betas_tmp = np.load(afolder + '/betas__ModelInds--%i.npy'%aind)
                    betas[aind,:] = betas_tmp
                np.save(afolder + '/betas.npy',betas)
                print('Betas saved for', afolder)
                for aFN in beta_files:
                    os.remove(aFN)
                print('Betas deleted for', afolder)
                print()
