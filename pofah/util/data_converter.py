import numpy as np

def nan_or_inf_idx(array):
	idx = []
	idx.extend(np.argwhere(np.isnan(array))[:,0]) # events that have at least one particle with a nan value
	idx.extend(np.argwhere(np.isinf(array))[:,0]) # events that have at least one particle with an inf value
	return np.unique(np.asarray(idx))

def delete_nan_and_inf_events(constituents, features):
	idx = nan_or_inf_idx(constituents)
	print('[converter.xyze_to_eppt]: {} NaN or inf values found. deleting affected events'.format(len(idx)))
	return np.delete(constituents, idx, axis=0), np.delete(features, idx, axis=0)

def xyze_to_eppt(constituents):
	''' converts an array [N x 2 x 100, 4] of particles
		from px, py, pz, E to eta, phi, pt (mass omitted)
	'''
	PX, PY, PZ, E = range(4)
	pt = np.sqrt(np.float_power(constituents[:,:,:,PX], 2) + np.float_power(constituents[:,:,:,PY], 2), dtype='float16') # numpy.float16 dtype -> float power to avoid overflow
	eta = np.arcsinh(np.divide(constituents[:,:,:,PZ], pt, out=np.zeros_like(pt), where=pt!=0.), dtype='float16')
	phi = np.arctan2(constituents[:,:,:,PY], constituents[:,:,:,PX], dtype='float16')

	return np.stack([eta, phi, pt], axis=3)

def eppt_to_xyz(constituents):
	''' converts an array [N x 2 x 100, 4] of particles
		from eta, phi, pt to px, py, pz (energy omitted)
	'''
	ETA, PHI, PT = range(3)
	px = constituents[:,:,:,PT] * np.cos(constituents[:,:,:,PHI])
	py = constituents[:,:,:,PT] * np.sin(constituents[:,:,:,PHI])
	pz = constituents[:,:,:,PT] * np.sinh(constituents[:,:,:,ETA])

	return np.stack([px,py,pz], axis=3)
