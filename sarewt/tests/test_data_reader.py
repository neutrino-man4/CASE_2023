import unittest
import os
import numpy as np
import sarewt.data_reader as dare



class DataReaderTestCase(unittest.TestCase):

	def setUp(self):
		self.dir_path = '/eos/user/k/kiwoznia/data/VAE_data/baby_events/qcd_sqrtshatTeV_13TeV_PU40'
		self.reader = dare.DataReader(self.dir_path)
		self.file_paths = fnames = [os.path.join(self.dir_path, 'qcd_sqrtshatTeV_13TeV_PU40_'+str(n)+'.h5') for n in [34,721,96]]
		self.total_num_events_in_dir = 57964 + 36 + 58096


	# test reading events from single file
	@unittest.skip
	def test_read_events_from_file_no_cuts(self):
		constituents, features = self.reader.read_events_from_file(self.file_paths[0])
		self.assertEqual(constituents.shape, (57964, 2, 100, 3))
		self.assertEqual(features.shape, (57964, 11))
		# mean pt, dPhi, dEta
		self.assertAlmostEqual(np.mean(constituents[...,-1]), 4.276516632181256, places=4)
		self.assertAlmostEqual(np.mean(constituents[...,-2]), -2.0955217934427573e-05, places=4)
		self.assertAlmostEqual(np.mean(constituents[...,-3]), -0.00046528394023909826)
		# limits dEta, dPhi 
		self.assertAlmostEqual(np.min(constituents[...,-2]), -0.7999, places=3)
		self.assertAlmostEqual(np.min(constituents[...,-3]), -0.7999, places=3)
		self.assertAlmostEqual(np.max(constituents[...,-2]), 0.7999, places=3)
		self.assertAlmostEqual(np.max(constituents[...,-3]), 0.7999, places=3)
		# mean mjj, dEtaJJ, dPhiJJ
		self.assertAlmostEqual(np.mean(features[:,0]), 741.7274080953777, places=3)
		self.assertAlmostEqual(np.mean(features[:,-2]), -0.0025248032581180646, places=3)
		self.assertAlmostEqual(np.mean(features[:,-1]), 0.002353363388548727, places=3)


	# test read events from dir no cuts
	def test_read_events_from_dir_number(self):

		# check reading with no limit on number
		constituents, constituents_names, features, features_names = self.reader.read_events_from_dir()
		self.assertEqual(len(features), self.total_num_events_in_dir)
		self.assertEqual(len(constituents), len(features))
		self.assertEqual(len(constituents_names), 3)
		self.assertEqual(len(features_names), 11)
			# check shape of events read
		self.assertEqual(len(constituents.shape), 4)
		self.assertEqual(constituents.shape[1], 2) # 2 jets
		self.assertEqual(constituents.shape[2], 100) # 100 particles
		self.assertEqual(constituents.shape[3], 3) # eta, phi, pt
		self.assertEqual(len(features.shape), 2)
		self.assertEqual(features.shape[1], 11) # 11 features

		# check reading with limit on number
		num_samples_to_read = 77000
		constituents, constituents_names, features, features_names = self.reader.read_events_from_dir(read_n=num_samples_to_read)
		self.assertEqual(len(features), num_samples_to_read)
		self.assertEqual(len(constituents), len(features))
		self.assertEqual(len(constituents_names), 3)
		self.assertEqual(len(features_names), 11)

			# check shape of events read
		self.assertEqual(len(constituents.shape), 4)
		self.assertEqual(constituents.shape[1], 2) # 2 jets
		self.assertEqual(constituents.shape[2], 100) # 100 particles
		self.assertEqual(constituents.shape[3], 3) # eta, phi, pt
		self.assertEqual(len(features.shape), 2)
		self.assertEqual(features.shape[1], 11) # 11 features


	# test read jet features from dir no cuts
	def test_read_jet_features_from_dir_number(self):

		# check reading with no limit on number
		features, features_names = self.reader.read_jet_features_from_dir(apply_mjj_cut=False)
		self.assertEqual(len(features), self.total_num_events_in_dir)
		self.assertEqual(len(features_names), 11)
			# check shape of events read
		self.assertEqual(len(features.shape), 2)
		self.assertEqual(features.shape[1], 11) # 11 features

		# check reading with limit on number
		num_samples_to_read = 66000
		features, features_names = self.reader.read_jet_features_from_dir(read_n=num_samples_to_read, apply_mjj_cut=False)
		self.assertEqual(len(features), num_samples_to_read)
		self.assertEqual(len(features_names), 11)

			# check shape of events read
		self.assertEqual(len(features.shape), 2)
		self.assertEqual(features.shape[1], 11) # 11 features


	# test event chunk generation from dir with num events splitting
	@unittest.skip
	def test_events_generated_by_num(self):

		read_events_n = 0
		parts_n = 1500
		
		for (constituents, features) in self.reader.generate_event_parts_from_dir(parts_n=parts_n):
		
			# check number of events read
			self.assertLessEqual(len(constituents), parts_n)
			self.assertEqual(len(constituents), len(features))
			self.assertGreater(len(constituents), 0)
			read_events_n += len(constituents)
		
			# check shape of events read
			self.assertEqual(len(constituents.shape), 4)
			self.assertEqual(constituents.shape[1], 2) # 2 jets
			self.assertEqual(constituents.shape[2], 100) # 100 particles
			self.assertEqual(constituents.shape[3], 3) # eta, phi, pt
			self.assertEqual(len(features.shape), 2)
			self.assertEqual(features.shape[1], 11) # 11 features


		# check that all events in dir have been read	
		self.assertEqual(read_events_n, self.total_num_events_in_dir)


	# test event chunk generation from dir with num events splitting
	@unittest.skip
	def test_constituents_generated_by_num(self):

		read_events_n = 0
		parts_n = 2500
		
		for constituents in self.reader.generate_constituents_parts_from_dir(parts_n=parts_n):
		
			# check number of events read
			self.assertLessEqual(len(constituents), parts_n)
			self.assertGreater(len(constituents), 0)
			read_events_n += len(constituents)
		
			# check shape of events read
			self.assertEqual(len(constituents.shape), 4)
			self.assertEqual(constituents.shape[1], 2) # 2 jets
			self.assertEqual(constituents.shape[2], 100) # 100 particles
			self.assertEqual(constituents.shape[3], 3) # eta, phi, pt

		# check that all events in dir have been read	
		self.assertEqual(read_events_n, 57964+36+58096)


	# test event chunk generation from dir with size events splitting 
	# TODO: implement
	@unittest.skip
	def test_events_generated_by_size(self):
		pass


	# test applying cuts to event sets
	# @unittest.skip
	def test_make_cuts(self):

		idx_mjj, idx_j1Pt, idx_j2Pt, idx_j1Eta, idx_dEta, idx_dPhi = 0, 1, 6, 2, 9, 10

		before_cut_n = 14
		constituents = np.ones((before_cut_n,100,3))
		features = np.ones((before_cut_n,12))
		# check cutting on mJJ
		features[:, idx_mjj] = [1500, 2000, 3000, 4, 5, 6, 1110, 2220, 9090, 10, 9876, 5623.3, 34.9, 2365.0] 
		cuts = {'mJJ': 1100}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum(features[:,idx_mjj]>cuts['mJJ']))
		self.assertEqual(len(features_after),len(const_after))

		# check cutting on sideband | dEta | > 1.4
		# no events in sideband
		features[0, idx_dEta] = 1.4
		cuts = {'sideband': 1.4}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),0)
		self.assertEqual(len(const_after),sum(np.abs(features[:,idx_dEta])>cuts['sideband']))
		self.assertEqual(len(features_after),len(const_after))
		# all events in signalregion
		cuts = {'signalregion': 1.4}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),len(constituents))
		self.assertEqual(len(const_after),sum(np.abs(features[:,idx_dEta])<=cuts['signalregion']))
		self.assertEqual(len(features_after),len(const_after))
		# mixed
		features[:, idx_dEta] = [-100, -0.5, 0, 3, 1.4, 0.001, 1.3, -1.4, 7236, -1.4, 200, 10, -12.3, 343]
		cuts = {'sideband': 1.4}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum(np.abs(features[:, idx_dEta])>cuts['sideband']))
		self.assertEqual(len(features_after),len(const_after)) 
		cuts = {'signalregion': 1.4}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum(np.abs(features[:, idx_dEta])<=cuts['signalregion']))
		self.assertEqual(len(features_after),len(const_after))

		# mJJ and dEta
		cuts = {'mJJ': 1100, 'sideband': 1.4}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum((features[:,idx_mjj]>cuts['mJJ'])*(np.abs(features[:, idx_dEta])>cuts['sideband'])))
		self.assertEqual(len(features_after),len(const_after))

		# add jetPt cuts
		# j1Pt
		features[:, idx_j1Pt] = [100, 200, 150, 300, 1.0, -100, 1234, 400, 12334.83, 1097.212, 0.01234, 34.45, 2, 8782]
		cuts = {'j1Pt': 200.}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum(features[:,idx_j1Pt]>cuts['j1Pt']))
		self.assertEqual(len(features_after),len(const_after))
		# j2Pt
		features[:, idx_j2Pt] = [1000, -3, 15, 300, 331.0, 0, 1., 899, 235.83, 1097.212, 999.9, 0.00202303, 1287.2, 212]
		cuts = {'j2Pt': 200.}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum(features[:,idx_j2Pt]>cuts['j2Pt']))
		self.assertEqual(len(features_after),len(const_after))
		# j1Pt & j2Pt
		cuts = {'j1Pt': 200., 'j2Pt': 200.}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum((features[:,idx_j1Pt]>cuts['j1Pt'])*(features[:,idx_j2Pt]>cuts['j2Pt'])))
		self.assertEqual(len(features_after),len(const_after))
		# j1Pt & j2Pt & mJJ
		cuts = {'mJJ': 1100, 'j1Pt': 200., 'j2Pt': 200.}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum((features[:,idx_j1Pt]>cuts['j1Pt'])*(features[:,idx_j2Pt]>cuts['j2Pt'])*(features[:,idx_mjj]>cuts['mJJ'])))
		self.assertEqual(len(features_after),len(const_after))
		# either ptJ1 or ptJ2
		cuts = {'jXPt': 200.}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum((features[:,idx_j1Pt]>cuts['jXPt'])+(features[:,idx_j2Pt]>cuts['jXPt'])))
		self.assertEqual(len(features_after),len(const_after))

		# jetEta cuts
		# j1eta
		features[:, idx_j1Eta] = [-1.123, 2.1, 150, 2.3, 1.0, -100, 1234, 0.4, 4.83, -1.212, 0.01234, 3.45, 2, 0.8782]
		cuts = {'j1Eta': 2.4}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum(np.abs(features[:,idx_j1Eta])<cuts['j1Eta']))
		self.assertEqual(len(features_after),len(const_after))
		cuts = {'j2Eta': 2.4}
		const_after, features_after = self.reader.make_cuts(constituents, features, **cuts)
		self.assertEqual(len(const_after),sum(np.abs(features[:, idx_dEta]+features[:,idx_j1Eta])<cuts['j2Eta']))
		self.assertEqual(len(features_after),len(const_after))
		



if __name__ == '__main__':
	unittest.main()
