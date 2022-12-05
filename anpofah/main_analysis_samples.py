import numpy as np
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_input as sdfi
import pofah.phase_space.cut_constants as cuts
import anpofah.sample_analysis.sample_analysis as saan


feature_analysis_all = False
feature_analysis_qcd = False
feature_analysis_sig = True
constituents_analysis = False

# samples and their paths
sample_ids_grav_35 = ['GtoWW35na','GtoWW35br']
sample_ids_grav = ['GtoWW15na','GtoWW15br','GtoWW25na','GtoWW25br','GtoWW35na','GtoWW35br','GtoWW45na','GtoWW45br',]
#sample_ids_azzz = ['AtoHZ15', 'AtoHZ20', 'AtoHZ25', 'AtoHZ30', 'AtoHZ35', 'AtoHZ40', 'AtoHZ45']
sample_ids_azzz = ['AtoHZ15', 'AtoHZ25', 'AtoHZ35', 'AtoHZ45']
sample_ids_qcd = ['qcdSide', 'qcdSideExt', 'qcdSig', 'qcdSigExt']
sample_ids_qcd_sb_vs_sr = ['qcdSide', 'qcdSig']
sample_ids_qcd_grav = sample_ids_qcd_sb_vs_sr + sample_ids_grav
sample_ids_qcd_azzz = sample_ids_qcd_sb_vs_sr + sample_ids_azzz
cuts = cuts.signalregion_cuts

paths = sf.SamplePathDirFactory(sdfi.path_dict)
fig_dir = 'fig/merged_data_for_VAE'
print('plotting to '+ fig_dir)

if feature_analysis_qcd:

	data_side = sf.read_inputs_to_event_sample_dict_from_dir(['qcdSide'], paths, read_n=int(1e6), **cuts.sideband_cuts)
	data_signalregion = sf.read_inputs_to_event_sample_dict_from_dir(['qcdSig'], paths, read_n=int(1e6), **cuts.signalregion_cuts)
	data = {**data_side, **data_signalregion}
	suffix = 'sb_vs_sr'

	# 1D distributions
	for feature in ['mJJ', 'DeltaEtaJJ', 'DeltaPhiJJ', 'j1Pt', 'j2Pt', 'j1Eta']:
		saan.analyze_feature(data, feature, plot_name=feature+'_qcd_'+suffix, fig_dir=fig_dir, first_is_bg=False, legend_loc='best', fig_format='.png')
	map_fun = lambda ff : ff['DeltaEtaJJ'] + ff['j1Eta'] # compute j2Eta
	saan.analyze_feature(data, 'j2Eta', map_fun=map_fun, plot_name='j2Eta_new_qcd_'+suffix, fig_dir=fig_dir, first_is_bg=False, legend_loc='best', fig_format='.png')

	# 2D distributions
	saan.analyze_feature_2D(data, 'j1Pt', 'j1Eta', fig_dir=fig_dir)
	saan.analyze_feature_2D(data, 'j2Pt', 'j2Eta', map_fun_2=map_fun, fig_dir=fig_dir)


if feature_analysis_all:

	for (sample_ids, suffix) in zip([sample_ids_qcd_sb_vs_sr, sample_ids_qcd_grav, sample_ids_qcd_azzz, sample_ids_qcd],['sb_vs_sr', 'vs_grav', 'vs_azzz', 'sb_vs_sr_ext']):

		# read 
		data = sf.read_inputs_to_event_sample_dict_from_dir(sample_ids, paths, read_n=int(1e6), mJJ=1200.)

		for feature in ['mJJ', 'DeltaEtaJJ', 'DeltaPhiJJ']:
			saan.analyze_feature(data, feature, plot_name=feature+'_qcd_'+suffix, fig_dir=fig_dir, first_is_bg=True, legend_loc='best', fig_format='.png')

if feature_analysis_sig:

	data = sf.read_inputs_to_event_sample_dict_from_dir(sample_ids_grav_35, paths, read_n=int(1e6), **cuts)
	saan.analyze_feature(data, 'mJJ', plot_name='mJJ_grav_35_all_cuts', fig_dir=fig_dir, first_is_bg=False, legend_loc='best', fig_format='.png')


if constituents_analysis:

	# read 
	sample_ids = sample_ids_qcd + sample_ids_grav + sample_ids_azzz
	data = sf.read_inputs_to_event_sample_dict_from_dir(sample_ids, paths, read_n=int(1e6), mJJ=1200.)

	for sample_id in sample_ids:

		sample = data[sample_id]
		saan.analyze_constituents(sample, clip_outlier=True, fig_dir=fig_dir, fig_format='.png')
		#sample.convert_to_cartesian()
		#saan.analyze_constituents(sample, clip_outlier=True, plot_name_suffix='_cartesian', fig_format='.png')


