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
import numpy as np

def sample_variation_list_creator(samples,tags):
	new_samples=[]
	for s in samples:
		for tag in tags:
			new_samples.append(s+f'_{tag}')
	return new_samples


def u_index(key):
	sys_weights_names=['pdf_up', 'pdf_down', 'prefire_up', 'prefire_down', 'pileup_up', 'pileup_down', 'btag_up', 'btag_down', 
    'PS_ISR_up', 'PS_ISR_down', 'PS_FSR_up', 'PS_FSR_down', 'F_up', 'F_down', 'R_up', 'R_down', 'RF_up', 'RF_down', 'top_ptrw_up', 'top_ptrw_down']
	try:
		ind=sys_weights_names.index(key)
	except:
		ind=-1
	return ind

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

data_BG_sample = samp.BG_SROrig_sample


# SIG_samples = ['XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrowReco',\
# 	'XToYYprimeTo4Q_MX3000_MY170_MYprime25_narrowReco',\
#                'XToYYprimeTo4Q_MX3000_MY25_MYprime25_narrowReco'
# 	]

#SIG_samples =
SIG_samples=['XToYYprimeTo4Q_MX2000_MY80_MYprime170_narrowReco']


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

# set up analysis outputs 
experiment = ex.Experiment(MC_run_n).setup(model_analysis_dir=True)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
print('Running analysis on experiment {}, plotting results to {}'.format(MC_run_n, experiment.model_analysis_dir))

d_experiment = ex.Experiment(data_run_n).setup(model_analysis_dir=True)
d_paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': d_experiment.run_dir})
# read in data


MC = sf.read_inputs_to_jet_sample_dict_from_dir([BGOrig_sample,BG_sample]+SIG_samples, paths)
DATA = sf.read_inputs_to_jet_sample_dict_from_dir([BGOrig_sample]+SIG_samples, d_paths)

#DATA_SCALED = sf.read_inputs_to_jet_sample_dict_from_dir_with_JE_tags(SIG_samples, d_paths,JE_tags)
MC_SCALED = sf.read_inputs_to_jet_sample_dict_from_dir_with_JE_tags(SIG_samples, paths,JE_tags)
MC.update(MC_SCALED)
#DATA.update(DATA_SCALED)
batch_n=256
# *****************************************
#					ROC
# *****************************************

fig = plt.figure()
fig.set_size_inches(13.,9.5)
alpha = 0.4
histtype = 'stepfilled'


plt.yscale('log')
ylabel='num frac'
xlabel='Max [(R J1 + 0.5*KL J1) & (R J2 + 0.5*KL J2)]'

plt.xlabel(xlabel)
plt.ylabel(ylabel)

sample_names=[BGOrig_sample]+SIG_samples+sample_variation_list_creator(SIG_samples,['nominal'])
d_sample_names=[BGOrig_sample]+SIG_samples#+sample_variation_list_creator(SIG_samples,['JES_up','JES_down'])

loss_combi_ids = ['rk5_05']

loss_strategy = lost.loss_strategy_dict['rk5_05']

bins=np.linspace(0,200,50)
legend=[DATA[s].name+'_DATA_VAE' for s in d_sample_names]

csv_dir='/work/abal/CASE/VAE_results/data_vs_mc/csv_files/'
f = open(csv_dir+f'counts_{SIG_samples[0]}.csv', 'w+')
writer = csv.writer(f)

for i, dat in enumerate([loss_strategy(DATA[s]) for s in d_sample_names]):
	if i > 0:
		histtype = 'step'
		alpha = 1.0
	else:
		writer.writerow(['legend','total evts.','% evts. surviving L>30','% evts. surviving L>50','DATA VAE'])
	idx = dpr.is_outlier_percentile(dat)
	#pdb.set_trace()
	#dat = dat[~idx]
	plt.hist(dat, bins=bins, density=True, alpha=alpha, histtype=histtype, label=legend[i])
	row = [legend[i],len(dat),len(dat[dat>30.])/len(dat)*100.,len(dat[dat>50.])/len(dat)*100.]
	writer.writerow(row)
#print(len)

weights=MC['weights']
sys_weights=np.empty(shape=(weights.shape[0],weights.shape[1]-1)) # Out of 21 columns, first column is nominal weight, rest are multiplicative factors
for i in range(1,weights.shape[1]):
	sys_weights[:,i-1]=weights[:,0]*weights[:,i]
sum_weights=np.zeros(shape=weights.shape[0])
for i in range(0,sys_weights.shape[1]):
	sum_weights[:]+=(sys_weights[:,i]-weights[:,0])**2
sum_weights=np.sqrt(sum_weights)

alpha=0.3
histtype = 'stepfilled'
legend=[MC[s].name+'_MC_VAE' for s in sample_names]
ucs=['PS_ISR_up','PS_ISR_down','PS_FSR_up','PS_FSR_down']
for i, dat in enumerate([loss_strategy(MC[s]) for s in sample_names]):
	#pdb.set_trace()
	if i > 0:
		histtype = 'step'
		alpha = 1.0
	else:
		writer.writerow([])
		writer.writerow(['legend','total evts.','% evts. surviving L>30','% evts. surviving L>50','MC VAE'])
	idx = dpr.is_outlier_percentile(dat)
	#dat = dat[~idx]
	if 'nominal' in legend[i]:
		#pdb.set_trace()
		#for uc in ucs: plt.hist(dat, bins=bins,weights=sys_weights[:,u_index(uc)], density=True, alpha=alpha, histtype=histtype, label=legend[i]+f'_{uc}')
		#plt.hist(dat, bins=bins,weights=sum_weights, density=True, alpha=alpha, histtype=histtype, label=legend[i]+'_quadrature')
		plt.hist(dat, bins=bins, density=True, alpha=alpha, histtype=histtype, label=legend[i]+'_nominal')
	else:
		plt.hist(dat, bins=bins,density=True, alpha=alpha, histtype=histtype, label=legend[i])
	row = [legend[i],len(dat),len(dat[dat>30.])/len(dat)*100.,len(dat[dat>50.])/len(dat)*100.]
	writer.writerow(row)

f.close()
plt.legend(loc='upper right')

web_dir='/etpwww/web/abal/public_html/CASE/Meetings/16-03-23/'
plt.savefig(web_dir+f'DATA_VS_MC_{SIG_samples[0]}rk5_05.png')
