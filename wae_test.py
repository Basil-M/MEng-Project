from __future__ import print_function

from os import mkdir
from os.path import exists
from shutil import rmtree

import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.callbacks import EarlyStopping, TerminateOnNaN as tnan

import utils

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

    params = utils.AttnParams()
    params["epochs"] = 15
    params["model_arch"] = "TRANSFORMER"
    params["layers"] = 1
    params["d_model"] = 3
    params["d_inner_hid"] = 4
    params["d_k"] = 4
    params["d_v"] = 4
    params["heads"] = 3
    params["pp_weight"] = 1.25

    # INTERIM DECODER
    params["bottleneck"] = "average"
    params["ID_width"] = 1
    params["latent_dim"] = 1
    params["stddev"] = 1
    params["dropout"] = 0.1
    MODEL_NAME = 'wae_test'
    MODELS_DIR = 'models/'
    MODEL_DIR = MODELS_DIR + MODEL_NAME + "/"

    # Process data
    params["data"] = "SIMULATED"
    params.setIDparams()
    tokens = utils.TokenList(['A', 'B'])
    dataStrsS = ['A', 'B', 'AB', 'BA', 'AA', 'BB', 'AAB', 'BAA', 'ABA', 'AAA', 'BBB', 'ABB', 'BAB', 'BBA', 'AAAA',
                 'BBBB', 'ABBB', 'BABB', 'BBAB', 'BBBA', 'AABB', 'ABAB', 'ABBA', 'BABA']

    # PARAMETERS TO CHOOSE
    ng = 6
    data_repetitions = 200000
    num_batches_per_epoch = 500
    w0done = False
    ng_vals = [23, 10, 3]
    w_vals = [0,10]
    pw_vals = [0, 5]
    vae_done = {}
    for ng in ng_vals:
        d = {}
        for w in w_vals:
            d2 = {}
            for p in pw_vals:
                d2[p] = False
            d[w] = d2
        vae_done[ng] = d

    for kernel in ["IMQ_normal"]:
        for ng in ng_vals:
            for nb in [50]:
                for s in [2]:
                    for w in w_vals:
                        for pw in pw_vals:
                        #     params["stddev"] = 1
                        #     # if w == 0 and not w0done:
                        #     #     params["stddev"] = 0
                        #     #     w0done = True
                        #     # elif w == 0 and w0done:
                        #         break
                            if w == 0:
                                params["stddev"] = 0
                            else:
                                params["stddev"] = 1

                            if ng == 1 and nb == 1 and kernel == "RBF":
                                # for some reason this fails so skib
                                break

                            # don't redo lots of VAEs
                            if kernel == "VAE":
                                print("VAE KERNEL: VAE_DONE[{}][{}][{}] =".format(ng,w, pw), vae_done[ng][w][pw])
                                if vae_done[ng][w][pw]:
                                    break
                                else:
                                    vae_done[ng][w][pw] = True

                            data_repetitions = 40000 * ng
                            num_per_batch = nb
                            params["WAE_s"] = s
                            params["kl_max_weight"] = w
                            params["WAE_kernel"] = None if kernel == "VAE" else kernel
                            params["pp_weight"] = pw
                            #
                            # num_per_batch = 25
                            # params["WAE_s"] = 2
                            # params["kl_max_weight"] = 10
                            params["epochs"] = int(np.ceil(data_repetitions / (num_per_batch * num_batches_per_epoch)))

                            ## PREPARE DATA
                            dataStrs = dataStrsS[:ng]
                            trainData = dd.SmilesToArray(dataStrs, tokens)
                            bs = num_batches_per_epoch * num_per_batch
                            bs = np.ceil(bs / (num_per_batch * ng)) * ng * num_per_batch
                            data_train = np.tile(trainData, (int(bs / ng), 1))
                            data_test = np.tile(trainData, (num_per_batch, 1))
                            params["batch_size"] = ng * num_per_batch

                            if params["pp_weight"]:
                                props = np.linspace(0, ng, len(dataStrs)) - ng/2
                                props = np.transpose(np.tile(props, (4, 1)))
                                props_train = np.tile(props, (int(bs / ng), 1))
                                props_test = np.tile(props, (num_per_batch, 1))
                                data_train = [data_train, props_train]
                                data_test = [data_test, props_test]

                            cb = 'potato' #EarlyStopping(monitor='val_acc', patience=10, verbose=0)
                            # Create model tracking folder
                            if exists(MODEL_DIR):
                                rmtree(MODEL_DIR)
                            mkdir(MODEL_DIR)
                            try:
                                model, results = trainTransformer(params=params, tokens=tokens, data_train=data_train,
                                                                  data_test=data_test,
                                                                  callbacks=["best_checkpoint", "var_anneal", cb,
                                                                             tnan()],
                                                                  model_dir=MODEL_DIR)
                            except:
                                print("training failed, what habben")
                            # try:
                            print("Autoencoder training complete. Loading best model.")
                            model.autoencoder.load_weights(MODEL_DIR + "best_model.h5")

                            # GET LATENTS
                            print("Generating latent representations from auto-encoder")
                            mu, logvar = model.encode.predict([trainData], ng)
                            var = np.exp(logvar)
                            # z_train = model.encode_sample.predict([data_train], 64)
                            # z_test = model.encode_sample.predict([data_test], 64)

                            ## PLOT THE DATA
                            gauss = lambda x, mu, var: np.exp(-0.5 * ((x - mu) ** 2) / var) / np.sqrt(
                                2 * np.pi * var)
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
                            sns.lineplot(x=x_vals, y=ysum, ax=ax,
                                         label="Total: m = {:.2f}, v = {:.2f})".format(mu_s, var_s))
                            print("Distribution over all data: mu = {}\tvar={}".format(mu_s, var_s))

                            plt.title("Val acc {:.3f}".format(np.amax(results.history['val_acc'])))
                            fig_name = "wae_test/wae_ng{}_npb{}_s{}_w{}_pw{}_k{}.png".format(ng, nb, s, w, pw,
                                                                                             kernel)
                            plt.savefig(fig_name, bbox_inches="tight")
                            plt.show()
                            plt.clf()

                            # print("Could not find file. Probably encounted nan on training :(")
#

if __name__ == '__main__':
    main()
