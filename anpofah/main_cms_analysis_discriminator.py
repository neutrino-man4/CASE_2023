import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr 
import pofah.util.sample_factory as sf
import anpofah.model_analysis.roc_analysis as ra
import anpofah.sample_analysis.sample_analysis as saan
import dadrah.selection.loss_strategy as lost
import anpofah.util.sample_names as samp
import pofah.util.config as co
import pdb
import subprocess
import h5py
import argparse
import json
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-s","--seed",type=int,default=12345,help="Set seed")
args = parser.parse_args()
run_n = args.seed

# setup analysis inputs
#do_analyses = ['roc', 'loss', 'roc_qcd_sb_vs_sr', 'loss_qcd_sb_vs_sr', 'loss_combi']
do_analyses = ['loss']
do_analyses = ['loss_combi']

fig_format = '.png'

# loss strategies
strategy_ids_total_loss = ['s5']
strategy_ids_reco_kl_loss = ['rk5_05']

strategy_ids_reco_loss = ['r5']
strategy_ids_kl_loss = ['kl5']


# set background sample to use
BG_sample = samp.BG_SR_sample
BGOrig_sample = samp.BG_SROrig_sample
SIG_QStar_samples = samp.SIG_QStar_samples
SIG_Graviton_samples = samp.SIG_Graviton_samples
SIG_Wkk_samples = samp.SIG_Wkk_samples
SIG_WpToBpT_samples = samp.SIG_WpToBpT_samples
SIG_XtoYY_samples = samp.SIG_XToYY_samples

mass_centers = samp.mass_centers
plot_name_suffix = BG_sample + '_vs_sig' 


# SIG_samples = ['XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrowReco',\
# 	'XToYYprimeTo4Q_MX3000_MY170_MYprime25_narrowReco',\
#                'XToYYprimeTo4Q_MX3000_MY25_MYprime25_narrowReco'
# 	]

SIG_samples = SIG_Graviton_samples+SIG_Wkk_samples+SIG_WpToBpT_samples
#SIG_samples =

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
experiment = ex.Experiment(run_n).setup(model_analysis_dir=True)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
print('Running analysis on experiment {}, plotting results to {}'.format(run_n, experiment.model_analysis_dir))
# read in data
data = sf.read_inputs_to_jet_sample_dict_from_dir(samp.all_samples, paths)

# with open(experiment.model_analysis_dir+"/params.json") as json_file: # Load parameters from JSON file
#     params=json.load(json_file)
#     batch_n=int(params['batch_n'])
batch_n=256
# *****************************************
#					ROC
# *****************************************
if 'roc' in do_analyses:
	# for each signal
	for SIG_sample, mass_center in zip(SIG_samples, mass_centers):
		# for each type of loss strategy
		#for loss_ids, loss_name in zip([strategy_ids_kl_loss, strategy_ids_reco_loss, strategy_ids_reco_kl_loss, strategy_ids_total_loss], ['KL_loss', 'reco_loss', 'reco_kl_loss', 'total_loss']):
		for loss_ids, loss_name in zip([strategy_ids_reco_kl_loss], ['reco_kl_loss']):
			# plot full ROC
			print(loss_ids)
			print(loss_name)
			print(SIG_sample)
			subprocess.call("mkdir -p %s"%(experiment.model_analysis_dir_roc+"/incl/"),shell=True)
			ra.plot_ROC_loss_strategy(data[BG_sample], data[SIG_sample], loss_ids, plot_name_suffix=plot_name_suffix+'_'+loss_name, fig_dir=experiment.model_analysis_dir_roc+"/incl/") 
			# plot binned ROC
			subprocess.call("mkdir -p %s"%(experiment.model_analysis_dir_roc+"/binned/"),shell=True)
			ra.plot_binned_ROC_loss_strategy(data[BG_sample], data[SIG_sample], mass_center, loss_ids, plot_name_suffix=plot_name_suffix+'_'+loss_name, fig_dir=experiment.model_analysis_dir_roc+"/binned/")


                        # I ADDED THIS BELOW
			aucs,tpr,fpr = ra.plot_binned_ROC_loss_strategy(data[BG_sample], data[SIG_sample], mass_center, loss_ids, plot_name_suffix=plot_name_suffix+'_'+loss_name, fig_dir=experiment.model_analysis_dir_roc+"/binned/")
			
			subprocess.call("mkdir -p %s"%(experiment.model_analysis_dir_roc+"/tpr_fpr_data/"),shell=True)
			h5f = h5py.File(experiment.model_analysis_dir_roc+"/tpr_fpr_data/"+BG_sample+"_"+SIG_sample+".h5", 'w') # store these datasets in the model analysis directory
			tp=h5f.create_dataset('tpr', data=tpr)
			fp=h5f.create_dataset('fpr', data=fpr)
			tp.attrs['AUC']=aucs # added attribute for storing calculated AUC score
			fp.attrs['AUC']=aucs # added attribute for storing calculated AUC score
			h5f.close()
			# Write AUC vs batch size data to a csv file
			auc_fields=[aucs[0],batch_n,run_n] # Place data in a list
			filepath = os.path.join(experiment.model_analysis_dir_aucdata,f"{SIG_sample}.csv")
			with open(filepath, 'a+') as f: # Open in append mode, create if it doesn't exist. 
				writer = csv.writer(f)
				writer.writerow(auc_fields)


if 'roc_qcd_mixed_vs_orig' in do_analyses:
	ra.plot_ROC_loss_strategy(data[samp.BG_SR_sample], data[samp.BG_SROrig_sample], strategy_ids_total_loss, plot_name_suffix=samp.BG_SR_sample + '_total_loss_mixed_vs_orig', fig_dir=experiment.model_analysis_dir_roc) 


# *****************************************
#			LOSS DISTRIBUTION
# *****************************************
if 'loss' in do_analyses:
	# plot loss distribution for qcd sig vs signals
	saan.analyze_feature(data, 'j1TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_QStar_samples, plot_name='loss_SR_QStar_TotalJ1_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j2TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_QStar_samples, plot_name='loss_SR_QStar_TotalJ2_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j1TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_XtoYY_samples, plot_name='loss_SR_XtoYY_TotalJ1_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j2TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_XtoYY_samples, plot_name='loss_SR_XtoYY_TotalJ2_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j1TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_Graviton_samples, plot_name='loss_SR_Graviton_TotalJ1_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j2TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_Graviton_samples, plot_name='loss_SR_Graviton_TotalJ2_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j1TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_Wkk_samples, plot_name='loss_SR_Wkk_TotalJ1_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j2TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_Wkk_samples, plot_name='loss_SR_Wkk_TotalJ2_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j1TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_WpToBpT_samples, plot_name='loss_SR_WpToBpT_TotalJ1_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j2TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_WpToBpT_samples, plot_name='loss_SR_WpToBpT_TotalJ2_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	#saan.analyze_feature(data, 'j1KlLoss', sample_names=[samp.BG_SR_sample]+SIG_samples, plot_name='loss_SR_distr_KL1_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	#saan.analyze_feature(data, 'j2KlLoss', sample_names=[samp.BG_SR_sample]+SIG_samples, plot_name='loss_SR_distr_KL2_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)

if 'loss_qcd_mixed_vs_orig' in do_analyses:
	#splot loss distribution for qcd side vs qcd signal region
	saan.analyze_feature(data, 'j1TotalLoss', sample_names=[samp.BG_SB_sample, samp.BG_SR_sample], plot_name='loss_distr_TotalJ1_qcdSB_vs_qcdSR', fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j2TotalLoss', sample_names=[samp.BG_SB_sample, samp.BG_SR_sample], plot_name='loss_distr_TotalJ2_qcdSB_vs_qcdSR', fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j1KlLoss', sample_names=[samp.BG_SB_sample, samp.BG_SR_sample], plot_name='loss_distr_KL1_qcdSB_vs_qcdSR', fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	saan.analyze_feature(data, 'j2KlLoss', sample_names=[samp.BG_SB_sample, samp.BG_SR_sample], plot_name='loss_distr_KL2_qcdSB_vs_qcdSR', fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)


# *****************************************
#			COMBINED LOSS DISTRIBUTION
# *****************************************
if 'loss_combi' in do_analyses:
	loss_combi_ids = ['rk5_05']
	for loss_id in loss_combi_ids:
		loss_strategy = lost.loss_strategy_dict[loss_id]
		# plot loss distribution for qcd mixed vs qcd original
		try:
			saan.analyze_feature(data, loss_strategy.title_str, map_fun=loss_strategy, sample_names=[samp.BG_SR_sample, samp.BG_SROrig_sample], plot_name='loss_distr_'+loss_strategy.file_str+'_qcd_mixed_vs_orig', fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
		except Exception as e:
			print('Something went wrong in plotting SR vs SB. Here is the error message.')
			print(e)
		#saan.analyze_feature(data, loss_strategy.title_str, map_fun=loss_strategy, sample_names=[samp.BG_SR_sample]+ samp.SIG_Wkk_samples, plot_name='loss_distr_Wkk_'+loss_strategy.file_str+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
			
		saan.analyze_feature(data, loss_strategy.title_str, map_fun=loss_strategy, sample_names=[samp.BG_SR_sample]+ samp.SIG_XToYY_samples[:9], plot_name='loss_distr_XtoYY_MX2000_'+loss_strategy.file_str+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
		saan.analyze_feature(data, loss_strategy.title_str, map_fun=loss_strategy, sample_names=[samp.BG_SR_sample]+ samp.SIG_XToYY_samples[9:20], plot_name='loss_distr_XtoYY_MX3000_'+loss_strategy.file_str+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
		saan.analyze_feature(data, loss_strategy.title_str, map_fun=loss_strategy, sample_names=[samp.BG_SR_sample]+ samp.SIG_XToYY_samples[20:], plot_name='loss_distr_XtoYY_MX5000_'+loss_strategy.file_str+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
		#saan.analyze_feature(data, loss_strategy.title_str, map_fun=loss_strategy, sample_names=[samp.BG_SR_sample]+ samp.SIG_QStar_samples, plot_name='loss_distr_QStar_'+loss_strategy.file_str+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
		#saan.analyze_feature(data, loss_strategy.title_str, map_fun=loss_strategy, sample_names=[samp.BG_SR_sample]+ samp.SIG_Graviton_samples, plot_name='loss_distr_QStar_'+loss_strategy.file_str+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
		


