import pofah.path_constants.sample_dict_file_parts_input as sdi
import pofah.phase_space.cut_constants as cuts
import pofah.util.sample_factory as sf
import numpy as np


def count_before_mjj_cut(data):

	with open('./data/event_counts_before_mjj_cuts.csv', 'a', newline='\n') as ff:

		ff.write(','.join(['sample_name', 'n_total', 'n_cut_mjj_1100', 'n_cut_dEta_1.4', 'n_cut_dEta_mjj', 'min_dEta', 'max_dEta', 'min_mJJ', 'max_mJJ']) + '\n\n')

		for sample_name, jet_events in data.items():

			# cut in mJJ and dEta
			jet_events_mjj_cut = jet_events.cut(jet_events['mJJ'] > 1100.) # cut on mJJ > 1100.
			jet_events_dEta_cut = jet_events.cut(np.abs(jet_events['DeltaEtaJJ']) > 1.4) # cut on |dEta| >= 1.4
			jet_events_mjj_dEta_cut = jet_events_mjj_cut.cut(np.abs(jet_events_mjj_cut['DeltaEtaJJ']) > 1.4) # cut mJJ *and* dEta
			
			# count jet_events
			n_total = len(jet_events)
			n_cut_mjj = len(jet_events_mjj_cut)
			n_cut_dEta = len(jet_events_dEta_cut)
			n_cut_mjj_dEta = len(jet_events_mjj_dEta_cut)

			with np.printoptions(precision=5, suppress=True):

				print("{: <12}: {: >7} n_total, {: >7} n_mjj_cut, {: >7} n_dEta_cut, {: >7} n_mjj_dEta_cut, {: >5} n_mjj_dEta_cut / n_total".format(sample_name, n_total, n_cut_mjj, n_cut_dEta, n_cut_mjj_dEta, n_cut_mjj_dEta/float(n_total)))

				ff.write(','.join([sample_name] + [str(n) for n in [n_total, n_cut_mjj, n_cut_dEta, n_cut_mjj_dEta, np.min(jet_events['DeltaEtaJJ']), np.max(jet_events['DeltaEtaJJ']), np.min(jet_events['mJJ']), np.max(jet_events['mJJ'])]]))
				ff.write('\n')


def count_after_mjj_cut(data, file_path):

	with open(file_path, 'a', newline='\n') as ff:

		ff.write(','.join(['sample_name', 'n_cut_mjj_1200', 'n_cut_dEta_mjj>1.4', 'min_dEta', 'max_dEta', 'min_mJJ', 'max_mJJ']) + '\n\n')

		for sample_name, jet_events in data.items():

			# cut in mJJ and dEta
			jet_events_dEta_cut = jet_events.cut(np.abs(jet_events['DeltaEtaJJ']) > 1.4) # cut on |dEta| >= 1.4
			
			# count jet_events
			n_cut_mjj = len(jet_events)
			n_cut_mjj_dEta = len(jet_events_dEta_cut)

			with np.printoptions(precision=5, suppress=True):

				print("{: <12}: {: >7} n_mjj_cut, {: >7} n_mjj_dEta_cut, {: >5} n_mjj_dEta_cut / n_cut_mjj".format(sample_name, n_cut_mjj, n_cut_mjj_dEta, n_cut_mjj_dEta/float(n_cut_mjj)))

				ff.write(','.join([sample_name] + [str(n) for n in [n_cut_mjj, n_cut_mjj_dEta, np.min(jet_events['DeltaEtaJJ']), np.max(jet_events['DeltaEtaJJ']), np.min(jet_events['mJJ']), np.max(jet_events['mJJ'])]]))
				ff.write('\n')


if __name__ == '__main__':

	full_mjj = False

	sample_ids = sdi.path_dict['sample_dir'].keys()
	paths = sf.SamplePathDirFactory(sdi.path_dict)
	data = sf.read_inputs_to_jet_sample_dict_from_dir(sample_ids, paths, **cuts.signalregion_cuts)
	file_path = './data/event_counts_after_mjj_1200_jetEta_2.4_jetPt_200_cut.csv'

	if full_mjj:
		count_before_mjj_cut(data, file_path)
	else:
		count_after_mjj_cut(data, file_path)
