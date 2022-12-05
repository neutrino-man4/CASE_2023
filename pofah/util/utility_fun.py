import operator
import numpy as np
import re


def filter_arrays_on_value(*arrays, filter_arr, filter_val, comp=operator.gt):
    idx_after_cut = comp(filter_arr,filter_val)
    return [a[idx_after_cut] for a in arrays]


def get_mean_and_stdev(dat): # nd.array [N,K,num-features]-> nd.array [num-features], nd.array [num-features]
	''' compute mean and std-dev of each feature (axis 2) of a datasample [N_examples, K_elements, F_features]
	'''
	std = np.nanstd(dat, axis=(0,1))
	mean = np.nanmean(dat, axis=(0,1))
	print('computed mean {} and std-dev {}'.format(mean, std))
	std[std == 0.0] = 0.001 # handle zeros
	return mean, std

def multi_replace(text, repl_dict):
	''' replace each key of dict with value of dict in text '''
	regex = re.compile("(%s)" % "|".join(map(re.escape,repl_dict.keys()))) 
	return regex.sub(lambda word: repl_dict[word.string[word.start():word.end()]], text)
