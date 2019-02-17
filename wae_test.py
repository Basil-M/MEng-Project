from __future__ import print_function

from os import mkdir
from os.path import exists
from shutil import rmtree

import numpy as np
import tensorflow as tf
from keras import backend as k

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))
import dataloader as dd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from train import trainTransformer

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
    MODEL_NAME = 'wae_test'
    MODELS_DIR = 'models/'
    MODEL_DIR = MODELS_DIR + MODEL_NAME + "/"

    # Create model tracking folder
    if exists(MODEL_DIR):
        rmtree(MODEL_DIR)
    mkdir(MODEL_DIR)

    # Process data
    params["d_file"] = "SIMULATED"
    params.setIDparams()
    tokens = dd.TokenList(['A', 'B'])
    dataStrs = ['A', 'B', 'AB', 'BA', 'AA', 'BB', 'AAB', 'BAA', 'ABA', 'AAA', 'BBB', 'ABB', 'BAB', 'BBA', 'AAAA',
                'BBBB', 'ABBB', 'BABB', 'BBAB', 'BBBA', 'AABB', 'ABAB', 'ABBA', 'BABA']

    # PARAMETERS TO CHOOSE
    ng = 6
    num_per_batch = 50
    num_batches_per_epoch = 500

    ## PREPARE DATA
    dataStrs = dataStrs[:ng]
    trainData = dd.SmilesToArray(dataStrs, tokens)
    bs = num_batches_per_epoch * num_per_batch
    bs = np.ceil(bs / (num_per_batch * ng)) * ng * num_per_batch
    data_train = np.tile(trainData, (int(bs / ng), 1))
    data_test = np.tile(trainData, (num_per_batch, 1))
    params["batch_size"] = ng * num_per_batch

    if params["pp_weight"]:
        props = np.linspace(0, 1, len(dataStrs))
        props = np.transpose(np.tile(props, (4, 1)))
        props_train = np.tile(props, (int(bs / ng), 1))
        props_test = np.tile(props, (num_per_batch, 1))
        data_train = [data_train, props_train]
        data_test = [data_test, props_test]

    model, results = trainTransformer(params=params, tokens=tokens, data_train=data_train, data_test=data_test,
                                      callbacks=["best_checkpoint", "var_anneal"], model_dir=MODEL_DIR)

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

    plt.title("Val accu {:.3f}".format(np.amax(results.history['val_accu'])))
    plt.show()


if __name__ == '__main__':
    main()
