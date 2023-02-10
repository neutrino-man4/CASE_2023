## Variational Autoencoders for the CMS Anomaly Search Effort (CASE)(MLG-23-002 and EXO-22-026): 

Don't bother trying to understand how all of it works. It just does. 

Be sure to install Python3, Tensorflow 2.X, h5py, scikit-learn and the usual packages. If something's missing, you will know anyways. 

### Getting Started
The paths are stored as key-value pairs in the form of Python dictionaries present in the following scripts (only necessary ones to get it all running are mentioned below):
- `pofah/path_constants/sample_dict.py`
    - reading input files for training/testing: `base_dir_events` 
- `pofah/path_constants/sample_dict_file_parts_input.py`
    - same as above: `base_dir`
- `pofah/path_constants/sample_dict_file_parts_reco.py`
    - writing the reconstructed files to disk: `base_dir` (these files are moderately large so watch your disk quota)
- `pofah/path_constants/experiment_dict.py` 
    - saving trained model: `model_dir`
    - rest should be self-explanatory. 


Once you are done setting the paths, move to `vande/`. Don't look at the other folders for now. 
