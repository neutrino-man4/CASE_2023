# VANDE
Variational Autoencoding for Anomaly Detection

### setup
requires Python3 and TF 2.X

## Particle VAE

### train
```
main_train_particle_vae.py
```

parameters
- run_n ... experiment number
- beta ... beta coefficient for Kullback-Leibler divergence term
- loss ... 3D+KL loss or MSE+KL loss (from losses module)
- reco_loss ... 3D or MSE (from losses module)
- cartesian ... True/False: constituents coordinates (if False: cylindrical)

### predict
```
main_predict_particle_vae.py
```
parameters
- run_n ... experiment number
- cartesian ... True/False: constituents coordinates (if False: cylindrical)

## Main CMS VAE

### Training
```
python3 main_cms_train_vae.py -s run_n -b batch_sz
```
Parameters:
- run_n: Experiment number, used as random seed
- batch_sz: Batch size

The VAE is trained on sideband jets which have been sampled in the Signal Region. The best trained model is saved at the path specified by `model_dir` in `pofah/experiment_dict.py` and its hyperparameters can be found in the `model_analysis_dir/params.json` file. 

 