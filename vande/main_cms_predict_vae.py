import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import setGPU
import tensorflow as tf
import numpy as np

tf.debugging.enable_check_numerics()

import pofah.util.event_sample as es
import vae.losses as losses
from vae.vae_particle import VAEparticle
import pofah.util.sample_factory as sf
import pofah.path_constants.sample_dict_file_parts_input as sdi 
import pofah.path_constants.sample_dict_file_parts_reco as sdr 
import sarewt.data_reader as dare
import pofah.phase_space.cut_constants as cuts
import pofah.util.experiment as expe
import training as train

import sys
import argparse
import json
# ********************************************************
#               runtime params
# ********************************************************

test_samples_Qstar_RS_W = ['QstarToQW_M_2000_mW_170',
                'QstarToQW_M_2000_mW_25',
                'QstarToQW_M_2000_mW_400',
                'QstarToQW_M_2000_mW_80',
                'QstarToQW_M_3000_mW_170',
                'QstarToQW_M_3000_mW_25',
                'QstarToQW_M_3000_mW_400',
                'QstarToQW_M_3000_mW_80',
                'QstarToQW_M_5000_mW_170',
                'QstarToQW_M_5000_mW_25',
                'QstarToQW_M_5000_mW_400',
                'QstarToQW_M_5000_mW_80',    
                'RSGravitonToGluonGluon_kMpl01_M_1000',
                'RSGravitonToGluonGluon_kMpl01_M_2000',
                'RSGravitonToGluonGluon_kMpl01_M_3000',
                'RSGravitonToGluonGluon_kMpl01_M_5000',
                'WkkToWRadionToWWW_M2000_Mr170',
                'WkkToWRadionToWWW_M2000_Mr400',
                'WkkToWRadionToWWW_M3000_Mr170',
                'WkkToWRadionToWWW_M3000_Mr400',
                'WkkToWRadionToWWW_M5000_Mr170',
                'WkkToWRadionToWWW_M5000_Mr400',
                'WpToBpT_Wp2000_Bp170_Top170_Zbt',
                'WpToBpT_Wp2000_Bp25_Top170_Zbt',
                'WpToBpT_Wp2000_Bp400_Top170_Zbt',
                'WpToBpT_Wp2000_Bp80_Top170_Zbt',
                'WpToBpT_Wp3000_Bp170_Top170_Zbt',
                'WpToBpT_Wp3000_Bp25_Top170_Zbt',
                'WpToBpT_Wp3000_Bp400_Top170_Zbt',
                'WpToBpT_Wp3000_Bp80_Top170_Zbt',
                'WpToBpT_Wp5000_Bp170_Top170_Zbt',
                'WpToBpT_Wp5000_Bp25_Top170_Zbt',
                'WpToBpT_Wp5000_Bp400_Top170_Zbt',
                'WpToBpT_Wp5000_Bp80_Top170_Zbt',
                'qcdSigTest',
                ]
test_samples_MC_ORIG = ['qcdSigMCOrig']

#test_samples = ['qcdSigQRTrain','qcdSigQRTest']
#test_samples = ['qcdSigMCOrig']
test_samples_X_TO_YY  = ['XToYYprimeTo4Q_MX2000_MY170_MYprime400_narrow',
                'XToYYprimeTo4Q_MX2000_MY25_MYprime25_narrow',
                'XToYYprimeTo4Q_MX2000_MY25_MYprime80_narrow',
                'XToYYprimeTo4Q_MX2000_MY400_MYprime170_narrow',
                'XToYYprimeTo4Q_MX2000_MY400_MYprime400_narrow',
                'XToYYprimeTo4Q_MX2000_MY400_MYprime80_narrow',
                'XToYYprimeTo4Q_MX2000_MY80_MYprime170_narrow',
                'XToYYprimeTo4Q_MX2000_MY80_MYprime400_narrow',
                'XToYYprimeTo4Q_MX2000_MY80_MYprime80_narrow',
                'XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrow',
                'XToYYprimeTo4Q_MX3000_MY170_MYprime25_narrow',
                'XToYYprimeTo4Q_MX3000_MY25_MYprime25_narrow',
                'XToYYprimeTo4Q_MX3000_MY25_MYprime400_narrow',
                'XToYYprimeTo4Q_MX3000_MY25_MYprime80_narrow',
                'XToYYprimeTo4Q_MX3000_MY400_MYprime170_narrow',
                'XToYYprimeTo4Q_MX3000_MY400_MYprime25_narrow',
                'XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow',
                'XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrow',
                'XToYYprimeTo4Q_MX3000_MY80_MYprime25_narrow',
                'XToYYprimeTo4Q_MX3000_MY80_MYprime400_narrow',
                'XToYYprimeTo4Q_MX5000_MY170_MYprime170_narrow',
                'XToYYprimeTo4Q_MX5000_MY170_MYprime80_narrow',
                'XToYYprimeTo4Q_MX5000_MY25_MYprime170_narrow',
                'XToYYprimeTo4Q_MX5000_MY25_MYprime25_narrow',
                'XToYYprimeTo4Q_MX5000_MY25_MYprime400_narrow',
                'XToYYprimeTo4Q_MX5000_MY25_MYprime80_narrow',
                'XToYYprimeTo4Q_MX5000_MY400_MYprime170_narrow',
                'XToYYprimeTo4Q_MX5000_MY400_MYprime25_narrow',
                'XToYYprimeTo4Q_MX5000_MY400_MYprime400_narrow',
                'XToYYprimeTo4Q_MX5000_MY80_MYprime25_narrow',
                'XToYYprimeTo4Q_MX5000_MY80_MYprime400_narrow'
  
]


#test_samples = ['qcdSig','qcdSideExt']
#test_samples = ['qcdSig', 'GtoWW35na']
#test_samples = ['qcdSideExt']
#test_samples = ['gravitonSig']
test_samples = test_samples_MC_ORIG + test_samples_X_TO_YY + test_samples_Qstar_RS_W 

parser = argparse.ArgumentParser()
parser.add_argument("-s","--seed",type=int,default=12345,help="Set seed")
args = parser.parse_args()



run_n = args.seed
cuts = cuts.sideband_cuts if 'qcdSideExt' in test_samples else cuts.signalregion_cuts #{}

experiment = expe.Experiment(run_n=run_n).setup(model_dir=True)
#batch_n = 1024

with open(experiment.model_analysis_dir_roc+"/tpr_fpr_data/params.json") as json_file: # Load parameters from JSON file
    params=json.load(json_file)
    batch_n=int(params['batch_n'])

print(os.path.join(experiment.model_dir, 'best_so_far'))	
# ********************************************
#               load model
# ********************************************

vae = VAEparticle.from_saved_model(path=os.path.join(experiment.model_dir, 'best_so_far'))
print('beta factor: ', vae.beta)
loss_fn = losses.threeD_loss


print(sdi.path_dict)
print(sdi.path_dict['sample_names'])
input_paths = sf.SamplePathDirFactory(sdi.path_dict)
print(input_paths)
result_paths = sf.SamplePathDirFactory(sdr.path_dict).update_base_path({'$run$': experiment.run_dir})

for sample_id in test_samples:

    # ********************************************
    #               read test data (events)
    # ********************************************


    list_ds = tf.data.Dataset.list_files(input_paths.sample_dir_path(sample_id)+'/*')
    print(list_ds)

    n_testsamples = 200

    #if 'Side' not in sample_id and 'qcd' in sample_id:
    #    n_testsamples = 8
    #if 'Side' in sample_id:
    #    n_testsamples = 1
        

    for file_path in list_ds.take(n_testsamples):
        #print("XXX")

        file_name = file_path.numpy().decode('utf-8').split(os.sep)[-1]

        #if 'bkg' not in file_name:
        #    continue
        #if 'batch19' not in file_name:
        #    continue

        print(file_name)

        test_sample = es.CaseEventSample.from_input_file(sample_id, file_path.numpy().decode('utf-8'), **cuts)
        test_evts_j1, test_evts_j2 = test_sample.get_particles()
        print('{}: {} j1 evts, {} j2 evts'.format(file_path.numpy().decode('utf-8'), len(test_evts_j1), len(test_evts_j2)))
        test_j1_ds = tf.data.Dataset.from_tensor_slices(test_evts_j1).batch(batch_n)
        test_j2_ds = tf.data.Dataset.from_tensor_slices(test_evts_j2).batch(batch_n)

        # *******************************************************
        #         forward pass test data -> reco and losses
        # *******************************************************
        
        print("HEREEE")
        print(sdi.path_dict['sample_names'])
        print(sample_id)
        print('predicting {}'.format(sdi.path_dict['sample_names'][sample_id]))
        reco_j1, loss_j1_reco, loss_j1_kl, orig_j1 = train.predict(vae.model, loss_fn, test_j1_ds)
        reco_j2, loss_j2_reco, loss_j2_kl, orig_j2 = train.predict(vae.model, loss_fn, test_j2_ds)

        #print("AAAAAAAAAAAAA test_j1_ds shape")
        #print(orig_j1.shape)

        z_mean1, z_log_var1, zs1 = train.predict_with_latent(vae.encoder, loss_fn, test_j1_ds)
        z_mean2, z_log_var2, zs2 = train.predict_with_latent(vae.encoder, loss_fn, test_j2_ds)

        #print("AAA")
        #print(z_mean1[2])
        #print("BBB")
        #print(z_log_var1[2])
        #print("CCC")
        #print(zs1[2])

        reco_j1_ptetaphi = np.array(reco_j1)
        reco_j2_ptetaphi = np.array(reco_j2)

        reco_j1_ptetaphi[...,0] = np.sqrt(reco_j1[:,:,0]**2+reco_j1[:,:,1]**2)
        reco_j2_ptetaphi[...,0] = np.sqrt(reco_j2[:,:,0]**2+reco_j2[:,:,1]**2)

        pt1 = np.array(reco_j1_ptetaphi[...,0])
        pt2 = np.array(reco_j2_ptetaphi[...,0])

        reco_j1_ptetaphi[:,:,1] = np.arcsinh(np.divide(np.array(reco_j1[:,:,2]),pt1, out=np.zeros_like(pt1), where=pt1!=0)) # eta = arcsinh(pz/pt)
        reco_j2_ptetaphi[:,:,1] = np.arcsinh(np.divide(np.array(reco_j2[:,:,2]),pt2, out=np.zeros_like(pt2), where=pt2!=0)) # eta = arcsinh(pz/pt)
        reco_j1_ptetaphi[:,:,2] = np.arccos(np.divide(np.array(reco_j1[:,:,1]),pt1, out=np.zeros_like(pt1), where=pt1!=0)) # eta = arcsinh(pz/pt)
        reco_j2_ptetaphi[:,:,2] = np.arccos(np.divide(np.array(reco_j2[:,:,1]),pt2, out=np.zeros_like(pt2), where=pt2!=0)) # eta = arcsinh(pz/pt)


        x_j1 = np.argsort(np.asarray(reco_j1_ptetaphi)[...,0]*(-1), axis=1)
        reco_j1 = np.take_along_axis(np.asarray(reco_j1_ptetaphi), x_j1[...,None], axis=1)        

        x_j2 = np.argsort(np.asarray(reco_j2_ptetaphi)[...,0]*(-1), axis=1)
        reco_j2 = np.take_along_axis(np.asarray(reco_j2_ptetaphi), x_j2[...,None], axis=1)        


        #print("YYYYYYYYYYYY")
        #print(orig_j1[1])

        orig_j1_ptetaphi = np.array(orig_j1)
        orig_j2_ptetaphi = np.array(orig_j2)

        orig_j1_ptetaphi[...,0] = np.sqrt(orig_j1[:,:,0]**2+orig_j1[:,:,1]**2)
        orig_j2_ptetaphi[...,0] = np.sqrt(orig_j2[:,:,0]**2+orig_j2[:,:,1]**2)

        pt1 = np.array(orig_j1_ptetaphi[...,0])
        pt2 = np.array(orig_j2_ptetaphi[...,0])

        orig_j1_ptetaphi[:,:,1] = np.arcsinh(np.divide(np.array(orig_j1[:,:,2]),pt1, out=np.zeros_like(pt1), where=pt1!=0)) # eta = arcsinh(pz/pt)
        orig_j2_ptetaphi[:,:,1] = np.arcsinh(np.divide(np.array(orig_j2[:,:,2]),pt2, out=np.zeros_like(pt2), where=pt2!=0)) # eta = arcsinh(pz/pt)
        orig_j1_ptetaphi[:,:,2] = np.arccos(np.divide(np.array(orig_j1[:,:,1]),pt1, out=np.zeros_like(pt1), where=pt1!=0)) # eta = arcsinh(pz/pt)
        orig_j2_ptetaphi[:,:,2] = np.arccos(np.divide(np.array(orig_j2[:,:,1]),pt2, out=np.zeros_like(pt2), where=pt2!=0)) # eta = arcsinh(pz/pt)


        orig_x_j1 = np.argsort(np.asarray(orig_j1_ptetaphi)[...,0]*(-1), axis=1)
        orig_j1 = np.take_along_axis(np.asarray(orig_j1_ptetaphi), orig_x_j1[...,None], axis=1)        

        orig_x_j2 = np.argsort(np.asarray(orig_j2_ptetaphi)[...,0]*(-1), axis=1)
        orig_j2 = np.take_along_axis(np.asarray(orig_j2_ptetaphi), orig_x_j2[...,None], axis=1)        



        #print(reco_j1[1])
        #print("XXXXXXXXXXX")
        #print(orig_j1[1])


        #losses_j1 = [losses.total_loss(loss_j1_reco, loss_j1_kl, vae.beta), loss_j1_reco, loss_j1_kl]
        #losses_j2 = [losses.total_loss(loss_j2_reco, loss_j2_kl, vae.beta), loss_j2_reco, loss_j2_kl]
        losses_j1 = [losses.total_loss(loss_j1_reco, loss_j1_kl, 0.5), loss_j1_reco, loss_j1_kl]
        losses_j2 = [losses.total_loss(loss_j2_reco, loss_j2_kl, 0.5), loss_j2_reco, loss_j2_kl]

        z_j1 = [z_mean1,z_log_var1,zs1]
        z_j2 = [z_mean2,z_log_var2,zs2]



        # *******************************************************
        #               add losses to DataSample and save
        # *******************************************************

        #reco_sample = es.CaseEventSample(sample_id + 'Reco', particles=[reco_j1, reco_j2], jet_features=test_sample.get_event_features(), particle_feature_names=test_sample.particle_feature_names)
        reco_sample = es.CaseEventSample(sample_id + 'Reco', particles=[reco_j1, reco_j2], jet_features=test_sample.get_event_features(), particle_feature_names=test_sample.particle_feature_names, orig_particles = [orig_j1, orig_j2])

        for loss, label in zip( losses_j1, ['j1TotalLoss', 'j1RecoLoss', 'j1KlLoss']):
            # import ipdb; ipdb.set_trace()    
            reco_sample.add_event_feature(label, loss)
        for loss, label in zip( losses_j2, ['j2TotalLoss', 'j2RecoLoss', 'j2KlLoss']):
            reco_sample.add_event_feature(label, loss)


        #for e, label in zip( [z_mean1], ['z_mean1_0']):
        #    print(len(np.array(e)))
        #    print(np.array(e).shape)
        #    reco_sample.add_event_feature(label, np.array(e)[:,0])



        #for e, label in zip( z_j2, ['z_mean2','z_log_var2','zs2']):
        #    reco_sample.add_event_feature(label, e)

        # *******************************************************
        #               write predicted data
        # *******************************************************
        print(sdr.path_dict['sample_names'])
        print('writing results for {} to {}'.format(sdr.path_dict['sample_names'][reco_sample.name], os.path.join(result_paths.sample_dir_path(reco_sample.name), file_name)))

        #reco_sample.dump(os.path.join(result_paths.sample_dir_path(reco_sample.name, mkdir=True), file_name))
        reco_sample.dump_with_orig(os.path.join(result_paths.sample_dir_path(reco_sample.name, mkdir=True), file_name))

