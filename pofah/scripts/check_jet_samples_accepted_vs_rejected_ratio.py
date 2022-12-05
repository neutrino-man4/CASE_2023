import sarewt.data_reader as dare
import pofah.jet_sample as jesa
import pofah.util.sample_factory as safa
import pofah.path_constants.sample_dict_file_parts_input as sadi
import os
import glob
import copy

path_dict = copy.deepcopy(sadi.path_dict)

base_dir = '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_results/run_101/sample_results/minRecoKL_loss'

quantiles = ['q1', 'q5', 'q10', 'q30', 'q50', 'q70', 'q90']
sample_ids = ['qcdSigAll', 'GtoWW15na', 'GtoWW25na', 'GtoWW35na', 'GtoWW45na']

for quantile in quantiles:
	path_dict.update(dict(base_dir=os.path.join(base_dir, quantile)))
	paths = safa.SamplePathDirFactory(path_dict)
	data = safa.read_inputs_to_jet_sample_dict_from_dir(sample_ids, paths)
	print('*'*10+'\n'+quantile+'\n'+'*'*10)
	for sample_id in sample_ids:
		accepted_n = len(data[sample_id].accepted())
		rejected_n = len(data[sample_id].rejected())
		print('{:<10}: {:>9} accepted, {:>9} rejetced. ratio acc/total: {:5f}'.format(sample_id, accepted_n, rejected_n, accepted_n/float(len(data[sample_id]))))
