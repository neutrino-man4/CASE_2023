import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import setGPU
import numpy as np
import random
import tensorflow as tf
from recordtype import recordtype
import pathlib
import copy
import sys
import json

import pandas as pd

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import dadrah.selection.discriminator as disc
import dadrah.selection.loss_strategy as lost
import dadrah.selection.qr_workflow as qrwf
import analysis.analysis_discriminator as andi
import dadrah.util.data_processing as dapr
import pofah.phase_space.cut_constants as cuts

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def inv_quantile_str(quantile):
    inv_quant = round((1.-quantile),2)
    return 'q{:02}'.format(int(inv_quant*100))

def fitted_selection(sample, strategy_id, polynomial):
    loss_strategy = lost.loss_strategy_dict[strategy_id]
    loss = loss_strategy(sample)
    loss_cut = polynomial
    return loss > loss_cut(sample['mJJ'])


#set_seeds(777)

#****************************************#
#           set runtime params
#****************************************#

iteration = sys.argv[1]

signal_contamin = { ('na', 0): [[0]]*4,
                    ('na', 100): [[1061], [1100], [1123], [1140]], # narrow signal. number of signal contamination; len(sig_in_training_nums) == len(signals)
                    ('br', 0): [[0]]*4,
                    ('br', 100): [[1065], [1094], [1113], [1125]], # broad signal. number of signal contamination; len(sig_in_training_nums) == len(signals)
                }


bin_edges = np.array([1200, 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206,
                        2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928,
                        4084, 4245, 4411, 4583, 4760, 4943, 5132, 5327, 5574, 5737, 5951, 6173, 6402, 6638, 6882]).astype('float')

bin_centers = [(high+low)/2 for low, high in zip(bin_edges[:-1], bin_edges[1:])]


# signals
resonance = 'na'
signals = ['grav_3p5_narrow']
masses = [3500]
xsecs = [0.]
sig_in_training_nums_arr = signal_contamin[(resonance, xsecs[0])] # TODO: adapt to multiple xsecs

# quantiles
quantiles = [0.3, 0.5, 0.7, 0.9, 0.95]
#quantiles = [0.3]
#quantiles = [0.1, 0.3]
regions = ["A","B","C","D","E"]
#quantiles = [0.5]

# to run
Parameters = recordtype('Parameters','run_n, qcd_sample_id, sig_sample_id, strategy_id, epochs, kfold, poly_order, read_n')
params = Parameters(run_n=113, 
                    qcd_sample_id='qcd',
                    sig_sample_id=None, # set sig id later in loop
                    strategy_id='rk5_05',
                    epochs=800,
                    kfold=5,
                    poly_order=6,
                    read_n=int(1e8))

result_dir = '/data/t3home000/bmaier/CASE/QR_results/analysis/run_%s/sig_WkkToWRadionToWWW_M3000_Mr170Reco/xsec_0/loss_rk5_05/qr_cuts/' % str(params.run_n) + '/maurizio_envelope'

subprocess.call("mkdir -p %s"%result_dir,shell=True)

#****************************************#
#           read in qcd data
#****************************************#
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n)})


cut_results = {}
polynomials = {}
spolynomials = {}
discriminators = {}

all_coeffs = {}

chunks = []
signal_samples = []

for k in range(params.kfold):
    
    # if datasets not yet prepared, prepare them, dump and return (same qcd train and testsample for all signals and all xsecs)
    qcd_train_sample, qcd_test_sample_ini = dapr.make_qcd_train_test_datasets(params, paths, which_fold=k, nfold=params.kfold, **cuts.signalregion_cuts)
    
    print("++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++")
    print(len(qcd_train_sample))
    print("++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++")
    
    
    # test sample corresponds to the other N-1 folds. It will not be used in the following.
    
    #****************************************#
    #      for each signal: QR
    #****************************************#
    
    for sig_sample_id, sig_in_training_nums, mass in zip(signals, sig_in_training_nums_arr, masses):
        
        params.sig_sample_id = sig_sample_id
        sig_sample_ini = js.JetSample.from_input_dir(params.sig_sample_id, paths.sample_dir_path(params.sig_sample_id), **cuts.signalregion_cuts)
        
        # ************************************************************
        #     for each signal xsec: train and apply QR
        # ************************************************************
        
        for xsec, sig_in_training_num in zip(xsecs, sig_in_training_nums):
            
            param_dict = {'$sig_name$': params.sig_sample_id, '$sig_xsec$': str(int(xsec)), '$loss_strat$': params.strategy_id}
            experiment = ex.Experiment(run_n=params.run_n, param_dict=param_dict).setup(model_dir_qr=True, analysis_dir_qr=True)
            #print(sdfs.path_dict)
            result_paths = sf.SamplePathDirFactory(sdfs.path_dict).update_base_path({'$run$': str(params.run_n), **param_dict}) # in selection paths new format with run_x, sig_x, ...
            
            # ************************************************************
            #                     train
            # ************************************************************
            
            # create new test samples for new xsec QR (quantile cut results)
            #qcd_test_sample = copy.deepcopy(qcd_test_sample_ini)
            
            sig_sample = copy.deepcopy(sig_sample_ini)
            mixed_train_sample, mixed_valid_sample = dapr.inject_signal(qcd_train_sample, sig_sample_ini, sig_in_training_num, train_split = 0.66)
            
            if k == 0:
                signal_samples.append(sig_sample)
                
                
            chunks.append(mixed_train_sample.merge(mixed_valid_sample))
                    
            # train QR model
            #discriminator = qrwf.train_LBSQR(quantile, mixed_train_sample, mixed_valid_sample, params)
            discriminator = qrwf.train_VQRv1(quantiles, mixed_train_sample, mixed_valid_sample, params)
        
            discriminators.update({"fold_%s"%str(k):discriminator})


bin_centers = [1200, 1300, 1474.1252, 1560.6403, 1694.2654, 1827.9368, 1961.5662, 2095.2969, 2228.7554, 2362.04, 2495.531, 2629.0693, 2762.6633, 2895.8464, 3030.178, 3163.2517, 3296.309, 3429.8882, 3563.1992, 3697.7837, 3828.7358, 3961.4727, 4099.5864, 4227.971, 4365.957, 4494.6973, 4632.1885, 4764.8906, 4893.6655, 5024.822]

cut_dict = {}


for k in range(0,params.kfold):
    for q,quantile in enumerate(quantiles):
        inv_quant = round((1.-quantile),2)
        qrcuts = np.empty([0, len(bin_centers)])
        counter = 0 
        for l in range(0,params.kfold):
            if k == l:
                continue

            yss = discriminators["fold_%s"%str(l)].predict(bin_centers)
            
            split_yss = np.array(np.split(np.array(yss),len(bin_centers)),dtype=object)[:,q]
            qrcuts = np.append(qrcuts, split_yss[np.newaxis,:], axis=0) 
        
        y_mean = np.mean(qrcuts,axis=0)

        cut_dict['{}_q{:02}'.format(str(k),int(inv_quant*100))] = y_mean
        #sel_q{:02}'.format(int(inv_quant*100))

    chunks[k].dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True),fold=k)

df = pd.DataFrame.from_dict(cut_dict)
df.to_csv("qrcuts.csv")

final_bkgsample = chunks[0]
for k in range(1,params.kfold):
    final_bkgsample = final_bkgsample.merge(chunks[k])

final_bkgsample.dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True))

for i,s in enumerate(signal_samples):
    s.dump(result_paths.sample_file_path(params.sig_sample_id))


'''

for k in range(params.kfold):
    chunk = chunks[k]
    for quantile in quantiles:
        inv_quant = round((1.-quantile),2)
        qrcuts = np.empty([0, len(bin_centers)])
        have_one = False
        for k2 in range(params.kfold):
            #if have_one:
            #    continue
            if k == k2:
                continue
            discriminator = discriminators[inv_quantile_str(quantile)+"_%s"%str(k2)]
            qrcuts_part = discriminator.predict(bin_centers)
            qrcuts = np.append(qrcuts, qrcuts_part[np.newaxis,:], axis=0)
            have_one = True

        x_all = np.ravel(bin_centers + np.zeros_like(qrcuts))
        y_all = np.ravel(qrcuts)

        model = make_pipeline(PolynomialFeatures(params.poly_order), Ridge(alpha=50, fit_intercept=False))

        _ = model.fit(x_all[:, None], y_all)
        
        print("XXX")
        print(_)
        print("XXX")

        ridge = model.named_steps['ridge']
        print(ridge.coef_)

        polynomial = np.poly1d(ridge.coef_.tolist()[::-1])
        #print("Polynomial")
        #print(polynomial)

        selection = fitted_selection(chunk, params.strategy_id, polynomial)
        chunk.add_feature('sel_q{:02}'.format(int(inv_quant*100)), selection)

    chunks[k] = chunk
    
    chunks[k].dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True),fold=k)




# Merging folds for bkg, dumping bkg and signals                                                                                               

final_bkgsample = chunks[0]
for k in range(1,params.kfold):
    final_bkgsample = final_bkgsample.merge(chunks[k])

final_bkgsample.dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True))






for quantile in quantiles:
    inv_quant = round((1.-quantile),2)
    qrcuts = np.empty([0, len(bin_centers)])
    for k in range(params.kfold):
        discriminator = discriminators[inv_quantile_str(quantile)+"_%s"%str(k)]
        qrcuts_part = discriminator.predict(bin_centers)
        qrcuts = np.append(qrcuts, qrcuts_part[np.newaxis,:], axis=0)

    x_all = np.ravel(bin_centers + np.zeros_like(qrcuts))
    y_all = np.ravel(qrcuts)
    
    model = make_pipeline(PolynomialFeatures(params.poly_order), Ridge(alpha=50, fit_intercept=False))

    _ = model.fit(x_all[:, None], y_all)
    
    ridge = model.named_steps['ridge']
    print(ridge.coef_)
    
    polynomial = np.poly1d(ridge.coef_.tolist()[::-1])

    for i,s in enumerate(signal_samples):
        selection = fitted_selection(s, params.strategy_id, polynomial)
        s.add_feature('sel_q{:02}'.format(int(inv_quant*100)), selection)

        signal_samples[i] = s
    

for i,s in enumerate(signal_samples):
    s.dump(result_paths.sample_file_path(params.sig_sample_id))
'''
