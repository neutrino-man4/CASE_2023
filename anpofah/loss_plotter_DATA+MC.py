import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr 
import pofah.util.sample_factory as sf
import anpofah.model_analysis.roc_analysis as ra
import anpofah.sample_analysis.sample_analysis as saan
import dadrah.selection.loss_strategy as lost
import anpofah.util.sample_names as samp
import pofah.util.config as co
import anpofah.util.data_preprocessing as dpr
import pdb
import subprocess
import h5py
import argparse
import json
import os
import csv
import matplotlib.pyplot as plt

def sample_variation_list_creator(samples,tags):
	new_samples=[]
	for s in samples:
		for tag in tags:
			new_samples.append(s+f'_{tag}')
	return new_samples


JE_tags=['JES_up','JES_down','JER_up','JER_down','JMS_up','JMS_down','JMR_up','JMR_down','nominal']

parser = argparse.ArgumentParser()
parser.add_argument("-rd","--data-seed",type=int,default=141098,help="Set seed")
parser.add_argument("-rMC","--MC-seed",type=int,default=50005,help="Set seed")
parser.add_argument("-d","--data",action='store_true',help="Set true if data")
args = parser.parse_args()
MC_run_n = args.MC_seed
data_run_n = args.data_seed

# setup analysis inputs
#do_analyses = ['roc', 'loss', 'roc_qcd_sb_vs_sr', 'loss_qcd_sb_vs_sr', 'loss_combi']
do_analyses = ['loss_combi']
#do_analyses = ['loss_qcd_sb_vs_sr']
#do_analyses = ['roc']
fig_format = '.png'

# loss strategies
strategy_ids_total_loss = ['s5']
strategy_ids_reco_kl_loss = ['rk5_05']

strategy_ids_reco_loss = ['r5']
strategy_ids_kl_loss = ['kl5']


# set background sample to use
BG_sample = samp.BG_SR_sample

#if args.data:
data_BG_sample = 'qcdSigDataTestReco'

BGOrig_sample = samp.BG_SROrig_sample
SIG_QStar_samples = samp.SIG_QStar_samples
SIG_Graviton_samples = samp.SIG_Graviton_samples
SIG_Wkk_samples = samp.SIG_Wkk_samples
SIG_WpToBpT_samples = samp.SIG_WpToBpT_samples
mass_centers = samp.mass_centers
plot_name_suffix = BG_sample + '_vs_sig' 


# SIG_samples = ['XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrowReco',\
# 	'XToYYprimeTo4Q_MX3000_MY170_MYprime25_narrowReco',\
#                'XToYYprimeTo4Q_MX3000_MY25_MYprime25_narrowReco'
# 	]

#SIG_samples =
SIG_samples=['QstarToQW_M_2000_mW_170Reco']


mass_centers = [3000]*len(SIG_samples)
mass_centers=[]
for s in SIG_samples:
	if '1000' in s:
		mass_centers.append(1000)
	elif '2000' in s:
		mass_centers.append(2000)
	elif '3000' in s:
		mass_centers.append(3000)
	elif '5000' in s:
		mass_centers.append(5000)
	else:
		mass_centers.append(3000)
print(len(mass_centers))
print(len(SIG_samples))
#SIG_samples = SIG_samples[:2]

print(BG_sample)
print(SIG_samples)

#exit(1)

# set up analysis outputs 
experiment = ex.Experiment(MC_run_n).setup(model_analysis_dir=True)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
print('Running analysis on experiment {}, plotting results to {}'.format(MC_run_n, experiment.model_analysis_dir))

d_experiment = ex.Experiment(data_run_n).setup(model_analysis_dir=True)
d_paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': d_experiment.run_dir})
# read in data


MC = sf.read_inputs_to_jet_sample_dict_from_dir([BG_sample]+SIG_samples, paths)
DATA = sf.read_inputs_to_jet_sample_dict_from_dir([data_BG_sample]+SIG_samples, d_paths)

DATA_SCALED = sf.read_inputs_to_jet_sample_dict_from_dir_with_JE_tags(SIG_samples, d_paths,JE_tags)
MC_SCALED = sf.read_inputs_to_jet_sample_dict_from_dir_with_JE_tags(SIG_samples, paths,JE_tags)
MC.update(MC_SCALED)
DATA.update(DATA_SCALED)
batch_n=256
# *****************************************
#					ROC
# *****************************************

fig = plt.figure()
fig.set_size_inches(10.,7.5)
alpha = 0.4
histtype = 'stepfilled'


plt.yscale('log')
ylabel='num frac'
xlabel='Min [(R J1 + 0.5*KL J1) & (R J2 + 0.5*KL J2)]'

plt.xlabel(xlabel)
plt.ylabel(ylabel)

sample_names=[BG_sample]+SIG_samples+sample_variation_list_creator(SIG_samples,['nominal'])
d_sample_names=[data_BG_sample]+SIG_samples+sample_variation_list_creator(SIG_samples,['nominal'])

loss_combi_ids = ['rk5_05']

loss_strategy = lost.loss_strategy_dict['rk5_05']

legend=[DATA[s].name+'_DATA_VAE' for s in d_sample_names]
for i, dat in enumerate([loss_strategy(DATA[s]) for s in d_sample_names]):
	if i > 0:
		histtype = 'step'
		alpha = 1.0
	idx = dpr.is_outlier_percentile(dat)
	dat = dat[~idx]
	plt.hist(dat, bins=100, density=True, alpha=alpha, histtype=histtype, label=legend[i])

alpha=0.3
histtype = 'stepfilled'
legend=[MC[s].name+'_MC_VAE' for s in sample_names]
for i, dat in enumerate([loss_strategy(MC[s]) for s in sample_names]):
	if i > 0:
		histtype = 'step'
		alpha = 1.0
	idx = dpr.is_outlier_percentile(dat)
	dat = dat[~idx]
	plt.hist(dat, bins=100, density=True, alpha=alpha, histtype=histtype, label=legend[i])

plt.legend(loc='upper right')

web_dir='/etpwww/web/abal/public_html/CASE/Meetings/16-03-23/'

plt.savefig(web_dir+'DATA_VS_MC_QSTAR.png')
