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


class epoch_track(Callback):
    def __init__(self, params, param_filename):
        self._params = params
        self._filename = param_filename

    def on_epoch_end(self, epoch, logs={}):
        self._params.set("current_epoch", self._params.get("current_epoch") + 1)
        self._params.save(self._filename)
        return

    def on_train_end(self, logs={}):
        if not self._params.get("ae_trained"):
            self._params.set("current_epoch", 1)
            self._params.set("ae_trained", True)
        self._params.save(self._filename)
        return

    def epoch(self):
        return self._params.get("current_epoch")


def main():
    np.random.seed(42)

    from molecules.model import TriTransformer as model_arch

    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

    print("Making smiles dict")

    params = dd.AttnParams()
    params.set("epochs", 15)
    params.set("model_arch", "TRANSFORMER")
    params.set("layers", 1)
    params.set("d_model", 6)
    params.set("d_inner_hid", 196)
    params.set("d_k", 4)
    params.set("d_v", 4)
    params.set("heads", 4)
    params.set("pp_weight", 1)

    # INTERIM DECODER
    params.set("bottleneck", "average")
    params.set("ID_width", 1)
    params.set("ID_layers", 1)
    params.set("WAE_kernel", "IMQ_normal")
    params.set("WAE_s", 1)
    params.set("kl_max_weight", 1)
    params.set("latent_dim", 1)
    params.set("stddev", 1)
    params.set("dropout", 0.1)
    MODEL_NAME = 'SIMTEST2'
    MODELS_DIR = 'models/'
    MODEL_DIR = MODELS_DIR + MODEL_NAME + "/"

    # Create model tracking folder
    if exists(MODEL_DIR):
        rmtree(MODEL_DIR)
    mkdir(MODEL_DIR)

    # Process data
    params.set("d_file", "SIMULATED")
    params.setIDparams()
    tokens = dd.TokenList(['A', 'B'])
    dataStrs = ['AAAAAA', 'BBBBBB', 'ABABAB']
    props = [-1, 0, 1]
    props = np.transpose(np.tile(props, (4,1)))

    trainData = dd.SmilesToArray(dataStrs, tokens)
    ng = len(trainData)
    num_per_batch = 20
    bs = 500 * num_per_batch
    bs = np.ceil(bs / (num_per_batch * ng)) * ng * num_per_batch
    print("trainData :", np.shape(trainData))
    data_train = np.tile(trainData, (int(bs / ng), 1))
    props_train = np.tile(props, (int(bs / ng), 1))
    print("tiled data :", np.shape(data_train))
    # data_train = np.repeat(trainData, repeats=bs/ng, axis=0)
    data_test = np.tile(trainData, (num_per_batch, 1))
    props_test = np.tile(props, (num_per_batch, 1))
    params.set("batch_size", ng * num_per_batch)
    from molecules.model import TriTransformer as model_arch

    # Set up model
    model = model_arch(tokens, params)

    # Learning rate scheduler
    lr_scheduler = LRSchedulerPerStep(params.get("d_model"),
                                      4000)

    # Model saver
    best_model_saver = ModelCheckpoint(MODEL_DIR + "best_model.h5", save_best_only=True,
                                       save_weights_only=True)

    # Learning rate scheduler
    callbacks = []
    callbacks.append(LRSchedulerPerStep(params.get("d_model"),
                                        4000))  # there is a warning that it is slow, however, it's ok.

    param_filename = MODEL_DIR + "params.pkl"
    model.build_models()
    model.compile_vae(Adam(0.001, 0.9, 0.98, epsilon=1e-9, clipnorm=1.0, clipvalue=0.5))

    wa = WeightAnnealer_epoch(model.kl_loss_var,
                              anneal_epochs=params.get("kl_anneal_epochs"),
                              max_val=params.get("kl_max_weight"),
                              init_epochs=params.get("kl_pretrain_epochs"))
    # Train model
    # Delete any existing datafiles containing latent representations
    if exists(MODEL_DIR + "latents.h5"):
        remove(MODEL_DIR + "latents.h5")

    model.autoencoder.fit([data_train, props_train], None,
                          batch_size=params.get("batch_size"),
                          epochs=params.get("epochs"),
                          validation_data=([data_test, props_test], None),
                          callbacks=[lr_scheduler, wa, best_model_saver],
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
    x_vals = np.linspace(-5, 5, 501)
    # plot the prior
    sns.set(rc={"lines.linewidth": 1.5})
    ax = sns.lineplot(x=x_vals, y=gauss(x_vals, 0, 1), label="Prior = N(0, 1)")
    # plot the data
    for idx in range(ng):
        mu_i = mu[idx][0]
        var_i = var[idx][0]
        sns.set(rc={"lines.linewidth": 0.75})
        print("Data point {}:\tmu = {}\tvar={}".format(dataStrs[idx], mu_i, var_i))
        ax = sns.lineplot(x=x_vals, y=gauss(x_vals, mu_i, var_i), ax=ax, dashes=True,
                          label="{} = N({:.2f},{:.2f})".format(dataStrs[idx], mu_i, var_i))

    # plot the sum
    mu_s = np.mean(mu)
    var_s = np.mean(var)
    sns.set(rc={"lines.linewidth": 1.5})
    sns.lineplot(x=x_vals, y=gauss(x_vals, mu_s, var_s), ax=ax, label="Total = N({:.2f},{:.2f})".format(mu_s, var_s))
    print("Distribution over all data: mu = {}\tvar={}".format(mu_s, var_s))
    plt.show()


if __name__ == '__main__':
    main()
