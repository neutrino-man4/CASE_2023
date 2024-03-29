import os
#import setGPU
import numpy as np
from collections import namedtuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
print('tensorflow version: ', tf.__version__)
tf.config.run_functions_eagerly(True)

#tf.debugging.enable_check_numerics()

from tensorflow.python import debug as tf_debug

import vae.vae_particle as vap
import vae.losses as losses
import pofah.path_constants.sample_dict_file_parts_input as sdi
import pofah.util.experiment as expe
import pofah.util.sample_factory as safa
import util.data_generator as dage
import sarewt.data_reader as dare
import pofah.phase_space.cut_constants as cuts
import training as tra

import random 
from loguru import logger
import sys
import argparse
import json

import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("-b","--batchsize",type=int,default=1024,help="Set Batch Size")
parser.add_argument("-s","--seed",type=int,default=12345,help="Set seed")
args = parser.parse_args()


BATCH_SIZE=args.batchsize

seed = args.seed

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

set_seeds(seed)

# ********************************************************
#       runtime params
# ********************************************************

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

Parameters = namedtuple('Parameters', 'run_n input_shape kernel_sz kernel_ini_n beta epochs train_total_n gen_part_n valid_total_n batch_n z_sz activation initializer learning_rate max_lr_decay lambda_reg,comments,data_or_MC')
params = Parameters(run_n=seed, 
                    input_shape=(100,3),
                    kernel_sz=(1,3), 
                    kernel_ini_n=12,
                    beta=0.5,
                    epochs=100,
                    train_total_n=int(4e6),
                    valid_total_n=int(1e6), 
                    gen_part_n=int(1e6),
                    batch_n=BATCH_SIZE,
                    z_sz=12,
                    activation='elu',
                    initializer='he_uniform',
                    learning_rate=0.001,
                    max_lr_decay=8, 
                    lambda_reg=0.01,
                    comments='j1Pt bug fixed',
                    data_or_MC='MC') # 'L1L2'

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)

with open(os.path.join(experiment.model_analysis_dir,"params.json"),'w+') as f:
    print(f"Dumping model parameters to JSON file")
    json.dump(params._asdict(),f,indent=4)    
#logger.add(f"{experiment.model_dir}/logfile.txt")

#
#  MC, batch 128: reco ~ 3.0
#  Data, batch 128: reco ~ 9.6
#  Data, batch 1024: reco ~ 6.0
# 

# ********************************************************
#       prepare training (generator) and validation data
# ********************************************************

# train (generator)
print('>>> Preparing training dataset generator')
print(paths.sample_dir_path('qcdSigMCTrain'))
data_train_generator = dage.CaseDataGenerator(path=paths.sample_dir_path('qcdSigMCTrain'), sample_part_n=params.gen_part_n, sample_max_n=params.train_total_n, **cuts.global_cuts) # generate 10 M jet samples
train_ds = tf.data.Dataset.from_generator(data_train_generator, output_types=tf.float32, output_shapes=params.input_shape).batch(params.batch_n, drop_remainder=True) # already shuffled

# validation (full tensor, 1M events -> 2M samples)                                                                          
print('>>> Preparing validation dataset')
print(paths.sample_dir_path('qcdSigMCTest'))
const_valid, _, features_valid, _, truth_valid = dare.CaseDataReader(path=paths.sample_dir_path('qcdSigMCTest')).read_events_from_dir(max_n=params.valid_total_n, **cuts.global_cuts)
data_valid = dage.events_to_input_samples(const_valid, features_valid)
valid_ds = tf.data.Dataset.from_tensor_slices(data_valid).batch(params.batch_n, drop_remainder=True)

# stats for normalization layer
mean_stdev = data_train_generator.get_mean_and_stdev()

print("Printing mean and stdev")
print(mean_stdev)

print("wow")

# *******************************************************
#                       training options
# *******************************************************

print('>>> Preparing optimizer')
optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
loss_fn = losses.threeD_loss

# *******************************************************
#                       build model
# *******************************************************

print('>>> Building model')
vae = vap.VAEparticle(input_shape=params.input_shape, z_sz=params.z_sz, kernel_ini_n=params.kernel_ini_n, kernel_sz=params.kernel_sz, activation=params.activation, initializer=params.initializer, beta=params.beta)
vae.build(mean_stdev)

# *******************************************************                                                                   #                       train and save                                                                                      # *******************************************************                                                                    
print('>>> Launching Training')
trainer = tra.Trainer(optimizer=optimizer, beta=params.beta, patience=3, min_delta=0.03, max_lr_decay=params.max_lr_decay, lambda_reg=params.lambda_reg, annealing=False, datalength=params.train_total_n, batchsize=params.batch_n, lr_decay_factor=0.3)
losses_reco, losses_valid = trainer.train(vae=vae, loss_fn=loss_fn, train_ds=train_ds, valid_ds=valid_ds, epochs=params.epochs, model_dir=experiment.model_dir)
tra.plot_training_results(losses_reco, losses_valid, experiment.fig_dir)

vae.save(path=experiment.model_dir)

with open(os.path.join(experiment.model_dir,'completion_log.txt'),'w') as f:
    from datetime import datetime; f.write(f'Training completed | {datetime.now()}')
