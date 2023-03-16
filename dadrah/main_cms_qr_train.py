import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import setGPU
import tensorflow as tf
from collections import namedtuple

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import dadrah.selection.discriminator as disc
import dadrah.selection.loss_strategy as lost
import analysis.analysis_discriminator as andi
import vande.training as train


print('tf version: ' + tf.__version__)

#****************************************#
#           set runtime params
#****************************************#
Parameters = namedtuple('Parameters','run_n, qcd_sample_id, quantile, strategy')
params = Parameters(run_n=98765, qcd_sample_id='qcdSigTestReco', quantile=0.9, strategy=lost.loss_strategy_dict['s5'])

#****************************************#
#           read in data
#****************************************#
experiment = ex.Experiment(params.run_n).setup(model_dir_qr=True,analysis_dir_qr=True)

#print(sdfr.path_dict)

paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})

#print("SSSS %s"%params.qcd_sample_id)
#print("TTTT %s"%paths.sample_dir_path(params.qcd_sample_id))

qcd_sig_sample = js.JetSample.from_input_dir(params.qcd_sample_id, paths.sample_dir_path(params.qcd_sample_id), read_n=int(1e7)) # can train on max 4M events (20% of qcd SR)
qcd_train, qcd_valid = js.split_jet_sample_train_test(qcd_sig_sample, 0.8)
print('training on {} events, validating on {}'.format(len(qcd_train), len(qcd_valid)))

#****************************************#
#       train quantile regression
#****************************************#

discriminator = disc.QRDiscriminator_KerasAPI(quantile=params.quantile, loss_strategy=params.strategy, batch_sz=256, epochs=30,  n_layers=5, n_nodes=50)
losses_train, losses_valid = discriminator.fit(qcd_train, qcd_valid)
print(discriminator.model.summary())
#train.plot_training_results(losses_train, losses_valid, '/home/bmaier/public_html/figs/case/kvae/qr/run_%s'%params.run_n)
discriminator.save(os.path.join(experiment.model_dir_qr,'model_q'+str(int(params.quantile*100))+'.h5'))

andi.analyze_discriminator_cut(discriminator, qcd_train, plot_name='discr_cut_qnt'+str(int(params.quantile*100)), fig_dir=experiment.analysis_dir_qr_cuts)

