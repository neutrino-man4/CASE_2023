from collections import namedtuple
import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import dadrah.selection.loss_strategy as lost
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import dadrah.selection.discriminator as disc
import dadrah.analysis.analysis_discriminator as andi
import anpofah.sample_analysis.sample_analysis as saan
import matplotlib.pyplot as plt
import os
import numpy as np


#****************************************#
#			set runtime params
#****************************************#
Parameters = namedtuple('Parameters','run_n, sm_sample_id, strategy')
params = Parameters(run_n=101, sm_sample_id='qcdSigAllReco', strategy=lost.loss_strategy_dict['rk5'])

#****************************************#
#			read in data
#****************************************#
experiment = ex.Experiment(params.run_n)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
qcd_sig_sample = js.JetSample.from_input_dir(params.sm_sample_id, paths.sample_dir_path(params.sm_sample_id))

bin_2000 = qcd_sig_sample[[(qcd_sig_sample['mJJ'] >= 2000.) & (qcd_sig_sample['mJJ'] <= 2050.)]]

bin2000_sample = js.JetSample(qcd_sig_sample.name, bin_2000, title=qcd_sig_sample.name + ' ' + str(2000 / 1000) + ' <= mJJ <= ' + str(2050 / 1000))
bin_counts, bin_edges = saan.analyze_feature({bin2000_sample.name: bin2000_sample}, params.strategy.title_str, map_fun=params.strategy, sample_names=[bin2000_sample.name], title_suffix=', N = '+str(len(bin2000_sample)), plot_name='loss_distr_qcdsig_2000_2050', fig_dir='fig', clip_outlier=True, first_is_bg=False, normed=False)

model_base_dir = '/eos/home-k/kiwoznia/dev/data_driven_anomaly_hunting/dadrah/models'

discriminator10 = disc.QRDiscriminator(quantile=0.1, loss_strategy=params.strategy)
discriminator10.load(os.path.join(model_base_dir, 'dnn_run_101_QRmodel_train_sz30pc_qnt10_20200912.h5'))
discriminator10.set_mean_var_input_output(qcd_sig_sample['mJJ'], discriminator10.loss_strategy(qcd_sig_sample))

discriminator90 = disc.QRDiscriminator(quantile=0.9, loss_strategy=params.strategy)
discriminator90.load(os.path.join(model_base_dir, 'dnn_run_101_QRmodel_train_sz30pc_qnt90_20200912.h5'))
discriminator90.set_mean_var_input_output(qcd_sig_sample['mJJ'], discriminator90.loss_strategy(qcd_sig_sample))

fig = plt.figure(figsize=(8, 8))
xs = np.arange(2000, 2050, 0.001*(2050-2000))
plt.plot(xs, discriminator10.predict( xs ) , '-', lw=2.5, label='cut Q10')
plt.plot(xs, discriminator90.predict( xs ) , '-', lw=2.5, label='cut Q90')
plt.ylabel('L1 & L2 > LT')
plt.xlabel('$M_{jj}$ [GeV]')
plt.title('QR cut q10 & q90, mJJ 2000-2050')
plt.legend(loc='best')
plt.draw()
fig.savefig(os.path.join('fig/cut_bin2000.png'), bbox_inches='tight')
plt.close(fig)

with np.printoptions(precision=3, suppress=True):
	print('{:10}\n {}'.format('cumulative: ', np.cumsum(bin_counts)/float(len(bin2000_sample))))
	print('{:10}\n {}'.format('edges: ', bin_edges))