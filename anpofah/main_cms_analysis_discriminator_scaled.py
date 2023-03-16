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

def sample_variation_list_creator(samples,tags):
	new_samples=[]
	for s in samples:
		for tag in tags:
			new_samples.append(s+f'_{tag}')
	return new_samples


JE_tags=['JES_up','JES_down','JER_up','JER_down','JMS_up','JMS_down','JMR_up','JMR_down','nominal']

parser = argparse.ArgumentParser()
parser.add_argument("-s","--seed",type=int,default=12345,help="Set seed")
parser.add_argument("-d","--data",action='store_true',help="Set true if data")
args = parser.parse_args()
run_n = args.seed

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

if args.data:
	BG_sample = 'qcdSigDataTestReco'

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

SIG_samples = SIG_Graviton_samples+SIG_Wkk_samples+SIG_WpToBpT_samples
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
experiment = ex.Experiment(run_n).setup(model_analysis_dir=True)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
print('Running analysis on experiment {}, plotting results to {}'.format(run_n, experiment.model_analysis_dir))
# read in data
data = sf.read_inputs_to_jet_sample_dict_from_dir_with_JE_tags(SIG_samples, paths,JE_tags)
data_unscaled = sf.read_inputs_to_jet_sample_dict_from_dir([BG_sample]+SIG_samples, paths)
data.update(data_unscaled) # Merge both
#import pdb;pdb.set_trace()
#with open(experiment.model_analysis_dir+"/params.json") as json_file: # Load parameters from JSON file
#    params=json.load(json_file)
#    batch_n=int(params['batch_n'])
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
			filepath = os.path.join(experiment.model_analysis_dir_roc,f"{SIG_sample}.csv")	
			import pdb;pdb.set_trace()
			f=open(filepath, 'a+') # Open in append mode, create if it doesn't exist. 
			for ind,tag in enumerate(JE_tags):
				writer = csv.writer(f) # Opening a csv file for writing AUC scores
				
				print(f'Now plotting loss with {tag}')
				incl_dir=os.path.join(experiment.model_analysis_dir_roc,"incl",tag)
				
				binned_dir=os.path.join(experiment.model_analysis_dir_roc,"binned",tag)
				auc_dir=os.path.join(experiment.model_analysis_dir_roc,"tpr_fpr_data",tag)
				subprocess.call("mkdir -p %s"%(incl_dir),shell=True)
				ra.plot_ROC_loss_strategy(data_unscaled[BG_sample], data[SIG_sample+f'_{tag}'], loss_ids, plot_name_suffix=plot_name_suffix+'_'+loss_name, fig_dir=incl_dir) 
				# plot binned ROC
				subprocess.call("mkdir -p %s"%(binned_dir),shell=True)
				ra.plot_binned_ROC_loss_strategy(data_unscaled[BG_sample], data[SIG_sample+f'_{tag}'], mass_center, loss_ids, plot_name_suffix=plot_name_suffix+'_'+loss_name, fig_dir=binned_dir)

				if ind==0:
					ra.plot_binned_ROC_loss_strategy(data_unscaled[BG_sample], data_unscaled[SIG_sample], mass_center, loss_ids, plot_name_suffix=plot_name_suffix+'_'+loss_name, fig_dir=experiment.model_analysis_dir_roc+"/binned/")
					ra.plot_ROC_loss_strategy(data_unscaled[BG_sample], data_unscaled[SIG_sample], loss_ids, plot_name_suffix=plot_name_suffix+'_'+loss_name, fig_dir=experiment.model_analysis_dir_roc+"/incl/") 
					aucs,tpr,fpr = ra.plot_binned_ROC_loss_strategy(data_unscaled[BG_sample], data_unscaled[SIG_sample], mass_center, loss_ids, plot_name_suffix=plot_name_suffix+'_'+loss_name, fig_dir=experiment.model_analysis_dir_roc+"/binned/")
					auc_fields=[aucs[0],batch_n,'not_scaled',run_n] # Place data in a list
					writer.writerow(auc_fields)
				
							# I ADDED THIS BELOW
				aucs,tpr,fpr = ra.plot_binned_ROC_loss_strategy(data_unscaled[BG_sample], data[SIG_sample+f'_{tag}'], mass_center, loss_ids, plot_name_suffix=plot_name_suffix+'_'+loss_name, fig_dir=binned_dir)
				#import pdb;pdb.set_trace()
				subprocess.call("mkdir -p %s"%(auc_dir),shell=True)
				h5f = h5py.File(auc_dir+BG_sample+"_"+SIG_sample+".h5", 'w') # store these datasets in the model analysis directory
				tp=h5f.create_dataset('tpr', data=tpr)
				fp=h5f.create_dataset('fpr', data=fpr)
				tp.attrs['AUC']=aucs # added attribute for storing calculated AUC score
				fp.attrs['AUC']=aucs # added attribute for storing calculated AUC score
				h5f.close()
				# Write AUC vs batch size data to a csv file
				auc_fields=[aucs[0],batch_n,tag,run_n] # Place data in a list
				writer.writerow(auc_fields)
			f.close()

if 'roc_qcd_mixed_vs_orig' in do_analyses:
	ra.plot_ROC_loss_strategy(data[samp.BG_SR_sample], data[samp.BG_SROrig_sample], strategy_ids_total_loss, plot_name_suffix=samp.BG_SR_sample + '_total_loss_mixed_vs_orig', fig_dir=experiment.model_analysis_dir_roc) 


# *****************************************
#			LOSS DISTRIBUTION
# *****************************************
if 'loss' in do_analyses:
	# plot loss distribution for qcd sig vs signals
	# saan.analyze_feature(data, 'j1TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_QStar_samples, plot_name='loss_SR_QStar_TotalJ1_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
	# saan.analyze_feature(data, 'j2TotalLoss', sample_names=[samp.BG_SR_sample]+SIG_QStar_samples, plot_name='loss_SR_QStar_TotalJ2_'+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
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
		#saan.analyze_feature(data, loss_strategy.title_str, map_fun=loss_strategy, sample_names=[samp.BG_SR_sample, samp.BG_SROrig_sample], plot_name='loss_distr_'+loss_strategy.file_str+'_qcd_mixed_vs_orig', fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
		#saan.analyze_feature(data, loss_strategy.title_str, map_fun=loss_strategy, sample_names=[samp.BG_SR_sample]+SIG_samples+sample_variation_list_creator(SIG_samples,JE_tags), plot_name='complete_loss_distr_QStar_'+loss_strategy.file_str+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)
		saan.analyze_feature(data, loss_strategy.title_str, map_fun=loss_strategy, sample_names=[BG_sample]+SIG_samples+sample_variation_list_creator(SIG_samples,['nominal']), plot_name='nominal_loss_distr_QStar_'+loss_strategy.file_str+plot_name_suffix, fig_dir=experiment.model_analysis_dir_loss, clip_outlier=True, fig_format=fig_format)



