import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr 
import pofah.util.sample_factory as sf
import anpofah.sample_analysis.sample_analysis as saan
import dadrah.selection.loss_strategy as lost


run_n = 106
fig_format = '.png'
experiment = ex.Experiment(run_n).setup(model_analysis_dir=True)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})

sample_ids = ['qcdSideExtReco', 'qcdSigReco']

# read in data
data = sf.read_inputs_to_jet_sample_dict_from_dir(sample_ids, paths, read_n=int(5e5))


# ****************************************************
#			2D MJJ vs LOSS DISTRIBUTION
# ****************************************************

print('plotting to '+ experiment.model_analysis_dir_loss)
saan.analyze_feature_2D(data, 'j1Pt', 'j1Eta', fig_dir=experiment.model_analysis_dir_loss)

saan.analyze_feature_2D(data, 'mJJ', 'j1TotalLoss', fig_dir=experiment.model_analysis_dir_loss)
saan.analyze_feature_2D(data, 'mJJ', 'j2TotalLoss', fig_dir=experiment.model_analysis_dir_loss)
saan.analyze_feature_2D(data, 'mJJ', 'minL1L2', map_fun_2=lost.loss_strategy_dict['s5'], fig_dir=experiment.model_analysis_dir_loss)

