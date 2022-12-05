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

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def inv_quantile_str(quantile):
    inv_quant = round((1.-quantile),2)
    return 'q{:02}'.format(int(inv_quant*100))


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
signals = ['WkkToWRadionToWWW_M3000_Mr170Reco']
masses = [3000]
xsecs = [0.]
sig_in_training_nums_arr = signal_contamin[(resonance, xsecs[0])] # TODO: adapt to multiple xsecs

# quantiles
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
regions = ["A","B","C","D","E"]
#quantiles = [0.5]

# to run
model_path_date = '20220505'

Parameters = recordtype('Parameters','run_n, qcd_sample_id, sig_sample_id, strategy_id, epochs, read_n, kfold')
params = Parameters(run_n=50000, 
                    qcd_sample_id='qcdSigMCOrigReco',
                    sig_sample_id=None, # set sig id later in loop
                    strategy_id='rk5_05',
                    epochs=800,
                    kfold=5,
                    read_n=int(1e8))

result_dir = '/data/t3home000/bmaier/CASE/QR_results/analysis/run_%s/sig_WkkToWRadionToWWW_M3000_Mr170Reco/xsec_0/loss_rk5_05/qr_cuts/' % str(params.run_n) + '/envelope'


#****************************************#
#           read in qcd data
#****************************************#
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n)})


cut_results = {}


for quantile in quantiles:
    # using inverted quantile because of dijet fit code
    inv_quant = round((1.-quantile),2)
    qrcuts = np.empty([0, len(bin_centers)])

    chunks = []

    for k in range(params.kfold):

        # if datasets not yet prepared, prepare them, dump and return (same qcd train and testsample for all signals and all xsecs)
        qcd_train_sample, qcd_test_sample_ini = dapr.make_qcd_train_test_datasets(params, paths, which_fold=k, nfold=params.kfold, **cuts.signalregion_cuts)
    

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
                result_paths = sf.SamplePathDirFactory(sdfs.path_dict).update_base_path({'$run$': str(params.run_n), **param_dict}) # in selection paths new format with run_x, sig_x, ...

                # ************************************************************
                #                     train
                # ************************************************************
                
                # create new test samples for new xsec QR (quantile cut results)
                qcd_test_sample = copy.deepcopy(qcd_test_sample_ini)
                sig_sample = copy.deepcopy(sig_sample_ini)
                mixed_train_sample, mixed_valid_sample = dapr.inject_signal(qcd_train_sample, sig_sample_ini, sig_in_training_num)

                if k == 0:
                    sig_sample.dump(result_paths.sample_file_path(params.sig_sample_id))

                chunks.append(mixed_train_sample.merge(mixed_valid_sample))
                
                model_paths = []
            
                # train and save QR model
                discriminator = qrwf.train_QR(quantile, mixed_train_sample, mixed_valid_sample, params)

                # ********************************************
                #               predict
                # ********************************************
                qrcuts_part = discriminator.predict(bin_centers)
                qrcuts = np.append(qrcuts, qrcuts_part[np.newaxis,:], axis=0)


    for k in range(params.kfold):

        tmp_qrcuts = np.delete(qrcuts, k, axis=0)# deleting the data for the specific quantile 
        mu = np.mean(tmp_qrcuts, axis=0)
        mi = np.min(tmp_qrcuts, axis=0)
        ma = np.max(tmp_qrcuts, axis=0)
        rmse = np.sqrt(np.mean(np.square(tmp_qrcuts-mu), axis=0))
        cuts_for_quantile = np.stack([bin_centers, mu, rmse, mi, ma], axis=1)
    
        cut_results.update({inv_quantile_str(quantile)+"_%s"%regions[k]: cuts_for_quantile.tolist()})

        chunks[k].dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True),fold=k)


with open(os.path.join(result_dir, 'cut_stats.json'), 'w') as ff:
    print("Dumping JSON to")
    print(os.path.join(result_dir, 'cut_stats.json'))
    json.dump(cut_results, ff)


'''

                print("Predicting qcd_sample")
                print(sig_sample)
                qcd_test_sample = qrwf.predict_QR(lambda_discriminator, qcd_test_sample, inv_quant)
                print("Predicting sig_sample")
                print(sig_sample)
                sig_sample = qrwf.predict_QR(lambda_discriminator, sig_sample, inv_quant)

            # write results for all quantiles
            print('writing selections to ', result_paths.base_dir)
            qcd_test_sample.dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True),fold=k)
            sig_sample.dump(result_paths.sample_file_path(params.sig_sample_id),fold=k)

'''

