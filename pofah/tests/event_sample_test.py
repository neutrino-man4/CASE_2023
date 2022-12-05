import pofah.util.sample_factory as sf
import pofah.path_constants.sample_dict_file_parts_input as sdfi
import pofah.util.event_sample as evsa


# *********************************************
#		unit testing event_sample
# *********************************************

def test_reading_from_dir():
	# test reading in event sample from dir
	name = 'qcdSig'
	path = '/eos/user/k/kiwoznia/data/VAE_data/baby_events/qcd_sqrtshatTeV_13TeV_PU40'

	sample = evsa.EventSample.from_input_dir(name, path)
	p1, p2 = sample.get_particles()

	assert p1.shape == p2.shape
	assert len(p1.shape) == 3


if __name__ == '__main__':
	test_reading_from_dir()
