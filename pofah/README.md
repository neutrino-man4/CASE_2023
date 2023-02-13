# pofah
Physics Objects for Anomaly Hunting

The input/output paths can be found here. 

These paths are stored as key-value pairs in the form of Python dictionaries present in the following scripts (only necessary ones to get it all running are mentioned below):
- `pofah/path_constants/sample_dict.py`
    - reading input files for training/testing: `base_dir_events` 
- `pofah/path_constants/sample_dict_file_parts_input.py`
    - same as above: `base_dir`
- `pofah/path_constants/sample_dict_file_parts_reco.py`
    - writing the reconstructed files to disk: `base_dir` (these files are moderately large so watch your disk quota)
- `pofah/path_constants/experiment_dict.py` 
    - saving trained model: `model_dir`
    - rest should be self-explanatory. 

The necessary directories for each run of the VAE are created by the `/util/experiment.py` script. In each script, you'll find an object of the form `experiment = expe.Experiment(model_dir=True,..)`. This calls the `Experiment` module within this class that creates the `model_dir` whose path is in the definition. All other directories can likewise be created, the possible options can be found in the `experiment.py` script as mentioned above. 