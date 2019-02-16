from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

from keras.optimizers import Adam
from keras.callbacks import Callback
from molecules.transformer import LRSchedulerPerStep, LRSchedulerPerEpoch
from os.path import exists
from os import mkdir, remove
from shutil import rmtree
import tensorflow as tf
from keras import backend as k

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))
import dataloader as dd
import matplotlib.pyplot as plt
from utils import WeightAnnealer_epoch
import seaborn as sns;

sns.set()

## extra imports to set GPU options
from tensorflow import ConfigProto, Session
from keras.backend.tensorflow_backend import set_session

###################################
# Prevent GPU pre-allocation
config = ConfigProto()
config.gpu_options.allow_growth = True
set_session(Session(config=config))


def main():
    np.random.seed(42)

    from molecules.model import TriTransformer as model_arch

    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

    print("Making smiles dict")

    params = dd.AttnParams()
    params["epochs"] = 15
    params["model_arch"] = "TRANSFORMER"
    params["layers"] = 3
    params["d_model"] = 3
    params["d_inner_hid"] = 4
    params["d_k"] = 4
    params["d_v"] = 4
    params["heads"] = 3
    params["pp_weight"] = 1.25

    # INTERIM DECODER
    params["bottleneck"] = "average"
    params["ID_width"] = 1
    params["WAE_kernel"] = "IMQ_normal"
    params["WAE_s"] = 2
    params["kl_max_weight"] = 10
    params["kl_pretrain_epochs"] = 1
    params["latent_dim"] = 1
    params["stddev"] = 1
    params["dropout"] = 0.1
    MODEL_NAME = 'SIMTEST2'
    MODELS_DIR = 'models/'
    MODEL_DIR = MODELS_DIR + MODEL_NAME + "/"

    # Create model tracking folder
    if exists(MODEL_DIR):
        rmtree(MODEL_DIR)
    mkdir(MODEL_DIR)

    # Process data
    params["d_file"] =  "SIMULATED"
    params.setIDparams()
    tokens = dd.TokenList(['A', 'B'])
    dataStrs = ['A', 'B', 'AB', 'BA', 'AA', 'BB', 'AAB', 'BAA', 'ABA', 'AAA', 'BBB', 'ABB', 'BAB', 'BBA','AAAA','BBBB','ABBB','BABB','BBAB','BBBA','AABB','ABAB','ABBA','BABA']

    ng = 6
    num_per_batch = 50
    num_batches_per_epoch= 500
    dataStrs = dataStrs[:ng]
    props = np.linspace(0, 1, len(dataStrs))
    props = np.transpose(np.tile(props, (4, 1)))
    trainData = dd.SmilesToArray(dataStrs, tokens)
    bs = num_batches_per_epoch* num_per_batch
    bs = np.ceil(bs / (num_per_batch * ng)) * ng * num_per_batch
    print("trainData :", np.shape(trainData))
    data_train = np.tile(trainData, (int(bs / ng), 1))
    props_train = np.tile(props, (int(bs / ng), 1))
    print("tiled data :", np.shape(data_train))
    # data_train = np.repeat(trainData, repeats=bs/ng, axis=0)
    data_test = np.tile(trainData, (num_per_batch, 1))
    props_test = np.tile(props, (num_per_batch, 1))
    params["batch_size"] =  ng * num_per_batch
    from molecules.model import TriTransformer as model_arch

    # Set up model
    model = model_arch(tokens, params)


    # Learning rate scheduler
    callbacks = []
    callbacks.append(LRSchedulerPerStep(params["d_model"],
                                        4000))  # there is a warning that it is slow, however, it's ok.

    # Model saver
    callbacks.append(ModelCheckpoint(MODEL_DIR + "best_model.h5", save_best_only=True,
                                       save_weights_only=True))

    model.build_models()
    model.compile_vae(Adam(0.001, 0.9, 0.98, epsilon=1e-9, clipnorm=1.0, clipvalue=0.5))

    callbacks.append(WeightAnnealer_epoch(model.kl_loss_var,
                              anneal_epochs=params["kl_anneal_epochs"],
                              max_val=params["kl_max_weight"],
                              init_epochs=params["kl_pretrain_epochs"]))
    # Train model
    # Delete any existing datafiles containing latent representations
    if exists(MODEL_DIR + "latents.h5"):
        remove(MODEL_DIR + "latents.h5")

    result = model.autoencoder.fit([data_train, props_train], None,
                                   batch_size=params["batch_size"],
                                   epochs=params["epochs"],
                                   validation_data=([data_test, props_test], None),
                                   callbacks=callbacks,
                                   shuffle=False)

    print("Autoencoder training complete. Loading best model.")
    model.autoencoder.load_weights(MODEL_DIR + "best_model.h5")

    # GET LATENTS
    print("Generating latent representations from auto-encoder")
    mu, logvar = model.encode.predict([data_test], ng)
    var = np.exp(logvar)
    # z_train = model.encode_sample.predict([data_train], 64)
    # z_test = model.encode_sample.predict([data_test], 64)

    ## PLOT THE DATA
    gauss = lambda x, mu, var: np.exp(-0.5 * ((x - mu) ** 2) / var) / np.sqrt(2 * np.pi * var)
    x_vals = np.linspace(-5, 5, 1001)
    # plot the prior
    sns.set(rc={"lines.linewidth": 1.5})
    ax = sns.lineplot(x=x_vals, y=gauss(x_vals, 0, 1), label="Prior = N(0, 1)")
    # plot the data
    ysum = np.zeros_like(x_vals)
    var_s = 0
    mu_s = 0
    for idx in range(ng):
        mu_i = mu[idx][0]
        var_i = var[idx][0]
        y_i = gauss(x_vals, mu_i, var_i)
        sns.set(rc={"lines.linewidth": 0.15})
        print("Data point {}:\tmu = {:.2f}\tvar={:.2f}".format(dataStrs[idx], mu_i, var_i))
        ax = sns.lineplot(x=x_vals, y=y_i, ax=ax, dashes=True,
                          label="{} = N({:.2f},{:.2f})".format(dataStrs[idx], mu_i, var_i))
        ysum += y_i / ng
        var_s += var_i / ng ** 2
        mu_s += mu_i / ng

    # plot the sum
    sns.set(rc={"lines.linewidth": 1.5})
    sns.lineplot(x=x_vals, y=ysum, ax=ax, label="Total: m = {:.2f}, v = {:.2f})".format(mu_s, var_s))
    print("Distribution over all data: mu = {}\tvar={}".format(mu_s, var_s))

    plt.title("Val accu {:.3f}".format(np.amax(result.history['val_accu'])))
    plt.show()

if __name__ == '__main__':
    main()
