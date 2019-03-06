from __future__ import print_function

import argparse
import os
from os import mkdir
from os.path import exists

import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam

import utils
from molecules.transformer import LRSchedulerPerStep
from utils import epoch_track, WeightAnnealer_epoch, load_dataset

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))

NUM_EPOCHS = 20
BATCH_SIZE = 50
LATENT_DIM = 128
RANDOM_SEED = 1403
DATA = 'data/zinc_1k.h5'
# DATA = 'C:\Code\MEng-Project\data\dummy2.txt'
# DATA = 'data/dummy.txt'
MODEL_ARCH = 'TRANSFORMER'
MODEL_NAME = 'test_ms'
MODEL_DIR = 'models/'

## extra imports to set GPU options
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow import ConfigProto, Session
from keras.backend.tensorflow_backend import set_session

###################################
# Prevent GPU pre-allocation
config = ConfigProto()
config.gpu_options.allow_growth = True
set_session(Session(config=config))


def get_arguments():
    default = utils.AttnParams()

    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--data', type=str, help='The HDF5 file containing preprocessed data.',
                        default=DATA)
    parser.add_argument('--model', type=str,
                        help='Name of the model - e.g. attn_dh128. The folder model_dir/model_name/ will contain saved models, parameters and tensorboard log-directories',
                        default=MODEL_NAME)
    parser.add_argument('--models_dir', type=str,
                        help='Path to folder containing model log directories e.g. model/',
                        default=MODEL_DIR)

    ### TRAINING PARAMETERS
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
                        help='Seed to use to start randomizer for shuffling.')
    parser.add_argument('--model_arch', type=str, metavar='N', default=MODEL_ARCH,
                        help='Model architecture to use - options are VAE, TRANSFORMER')
    parser.add_argument('--base_lr', type=float, metavar='0.001', default=0.001,
                        help='Base training rate for ADAM optimizer')
    parser.add_argument('--kl_pretrain_epochs', type=float, metavar='1', default=default["kl_pretrain_epochs"],
                        help='Number of epochs to train before introducing KL loss')
    parser.add_argument('--kl_anneal_epochs', type=float, metavar='5', default=default["kl_anneal_epochs"],
                        help='Number of epochs to anneal over')
    parser.add_argument('--kl_max_weight', type=float, metavar='1', default=default["kl_max_weight"],
                        help='Maximum KL weight')
    parser.add_argument('--WAE_kernel', type=str, metavar="IMQ_normal", default=default["WAE_kernel"],
                        help='Kernel for Wasserstein distance - options are RBF, IMQ_normal, IMQ_sphere and IMQ_uniform')
    parser.add_argument('--WAE_s', type=float, metavar='1', default=default["WAE_s"],
                        help='Scale factor to use for RBF for Wasserstein fitting. If 0, will use ELBO instead (default).')
    parser.add_argument('--pp_weight', type=float, metavar='1.5', default=default["pp_weight"],
                        help='For joint optimisation: Amount to weight MSE loss of property predictor')

    parser.add_argument('--gen', help='Whether to use generator for data', default=False)

    ### BOTTLENECK PARAMETERS
    parser.add_argument('--latent_dim', type=int, metavar='N', default=default["latent_dim"],
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--bottleneck', type=str, default=default["bottleneck"],
                        help='Bottleneck architecture - average, interim_decoder')

    parser.add_argument('--ID_d_model', type=int, metavar='N', default=None,
                        help='Dimensionality of interim decoder model')
    parser.add_argument('--ID_d_inner_hid', type=int, metavar='N', default=None,
                        help='Dimensionality of interim decoder fully connected networks after attention layers')
    parser.add_argument('--ID_d_k', type=int, metavar='N', default=None,
                        help='Dimensionality of interim decoder attention keys')
    parser.add_argument('--ID_d_v', type=int, metavar='N', default=None,
                        help='Dimensionality of interim decoder attention values')
    parser.add_argument('--ID_heads', type=int, metavar='N', default=None,
                        help='Number of interim decoder attention heads to use')
    parser.add_argument('--ID_layers', type=int, metavar='N', default=None,
                        help='Number of interim decoder layers')
    parser.add_argument('--ID_width', type=int, metavar='N', default=default["ID_width"],
                        help='Number of interim decoder layers')

    ### MODEL PARAMETERS
    parser.add_argument('--d_model', type=int, metavar='N', default=default["d_model"],
                        help='Dimensionality of transformer model')
    parser.add_argument('--d_inner_hid', type=int, metavar='N', default=default["d_inner_hid"],
                        help='Dimensionality of fully connected networks after attention layers')
    parser.add_argument('--d_k', type=int, metavar='N', default=default["d_k"],
                        help='Dimensionality of attention keys')
    parser.add_argument('--d_v', type=int, metavar='N', default=default["d_v"],
                        help='Dimensionality of attention values')
    parser.add_argument('--heads', type=int, metavar='N', default=default["heads"],
                        help='Number of attention heads to use')
    parser.add_argument('--layers', type=int, metavar='N', default=default["layers"],
                        help='Number of encoder/decoder layers')

    parser.add_argument('--dropout', type=float, metavar='0.1', default=default["dropout"],
                        help='Dropout to use in autoencoder')
    parser.add_argument('--stddev', type=float, metavar='0.01', default=default["stddev"],
                        help='Standard deviation of variational sampler')

    ### PROPERTY PREDICTOR PARAMETERS
    parser.add_argument('--pp_epochs', type=int, metavar='N', default=default["pp_epochs"],
                        help='Number of epochs to train property predictor')

    parser.add_argument('--pp_layers', type=int, metavar='N', default=default["pp_layers"],
                        help='Number of dense layers for property predictor')

    return parser.parse_args()


def trainTransformer(params, data_file=None, tokens=None, data_train=None, data_test=None, model_dir=None,
                     callbacks=None):
    if data_file:
        data_train, data_test, props_train, props_test, tokens = load_dataset(data_file, "TRANSFORMER",
                                                                              params["pp_weight"])
        if params["bottleneck"] == "GRU":
            data_train_onehot, data_test_onehot, _, _, _ = load_dataset(data_file, "GRU", False)
            data_train = [data_train_onehot, data_train]
            data_test = [data_test_onehot, data_test]
        else:
            data_train = [data_train, data_train]
            data_test = [data_test, data_test]

        # add properties
        if props_train is not None:
            data_train.append(props_train)
            data_test.append(props_test)

    if params["pp_weight"]:
        params["num_props"] = np.shape(data_test[2])[1]

    if callbacks is None:
        callbacks = ["checkpoint", "best_checkpoint", "tensorboard", "var_anneal", "epoch_track"]

    # GET NUMBER OF AVAILABLE GPUS
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES')
    if CUDA_VISIBLE_DEVICES is None:
        N_GPUS = 1
    else:
        N_GPUS = len(CUDA_VISIBLE_DEVICES.split(","))

    print("Found {} GPUs".format(N_GPUS))

    from molecules.model import TriTransformer as model_arch

    # Set up model
    if N_GPUS == 1:
        model = model_arch(tokens, params)
        model.build_models()
    else:
        # Want to set model up in the CPU
        with tf.device("/cpu:0"):
            model = model_arch(tokens, params)
            model.build_models()

    model.compile_vae(Adam(0.001, 0.9, 0.98, epsilon=1e-9, clipnorm=1.0, clipvalue=0.5), N_GPUS=N_GPUS)

    if params["current_epoch"] != 0 and model_dir is not None:
        if os.path.exists(model_dir + "model.h5"):
            model.autoencoder.load_weights(model_dir + "model.h5", by_name=True)
    # Store number of params
    params["num_params"] = model.autoencoder.count_params()

    # Set up callbacks
    # Learning rate scheduler

    model_trained = False
    n_pretrain = params["kl_pretrain_epochs"] + params["kl_anneal_epochs"]
    while not model_trained:
        pretraining_done = params["current_epoch"] > n_pretrain
        cb = []
        for c in callbacks:
            if not isinstance(c, str):
                cb.append(c)
            elif c == "checkpoint":
                cb.append(ModelCheckpoint(model_dir + "model.h5", save_best_only=False,
                                          save_weights_only=True))
            elif c == "best_checkpoint" and pretraining_done:
                # Best model saver
                # Don't include the best_checkpoint saver if the pretraining isn't done
                # During pretraining it is likely the model may have a better validation accuracy
                # Than when the variational objective is fully included
                cb.append(ModelCheckpoint(model_dir + "best_model.h5", save_best_only=True,
                                          save_weights_only=True))
            elif c == "tensorboard":
                # Tensorboard Callback
                cb.append(TensorBoard(log_dir=model_dir + "logdir/",
                                      histogram_freq=0,
                                      batch_size=params["batch_size"],
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='batch'))
            elif c == "var_anneal":
                cb.append(WeightAnnealer_epoch(model.kl_loss_var,
                                               anneal_epochs=params["kl_anneal_epochs"],
                                               max_val=params["kl_max_weight"],
                                               init_epochs=params["kl_pretrain_epochs"]))
            elif c == "epoch_track":
                callbacks.append(epoch_track(params, param_filename=model_dir + "params.pkl", csv_track=True))

        cb.append(LRSchedulerPerStep(params["d_model"],
                                     4000))  # there is a warning that it is slow, however, it's ok.


        results = model.autoencoder.fit(data_train, None, batch_size=params["batch_size"] * N_GPUS,
                                        epochs=params["epochs"] if pretraining_done else n_pretrain,
                                        initial_epoch=params["current_epoch"] - 1,
                                        validation_data=(data_test, None),
                                        callbacks=cb)
        params["current_epoch"] += len(results.history['acc'])
        model_trained = params["current_epoch"] > params["epochs"]

    return model, results


def main():
    args = get_arguments()
    np.random.seed(args.random_seed)

    model_dir = args.models_dir + args.model + "/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # Get default attention parameters
    params = utils.AttnParams()

    # Get training and test data from data file
    # Set up model
    for arg in vars(args):
        if arg in params.params:
            params[arg] = getattr(args, arg)

    ## VARIATIONAL AUTOENCODER
    if args.model_arch == "VAE":
        from molecules.model import MoleculeVAE
        data_train, data_test, tokens = load_dataset(args.data, "GRU", params["pp_weight"])
        if params["pp_weight"]:
            params["num_props"] = np.shape(data_test[1])[1]
            data_train_in = data_train[0]
            data_test_in = data_test[0]
        else:
            data_train_in = data_train
            data_test_in = data_test

        model = MoleculeVAE(tokens, params)
        checkpointer = ModelCheckpoint(filepath=model_dir + "best_model.h5", save_best_only=True,
                                       save_weights_only=True, monitor='val_acc')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      patience=3,
                                      min_lr=0.001)

        params.save(model_dir + "params.pkl")
        model.autoencoder.fit(
            data_train_in,
            data_train,
            shuffle=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[checkpointer, reduce_lr],
            validation_data=(data_test_in, data_test)
        )
    else:

        # Handle interim decoder parameters
        params.setIDparams()

        # Create model tracking folder
        if not exists(model_dir):
            mkdir(model_dir)

        # Handle parameters
        param_filename = model_dir + "params.pkl"
        loaded_params = utils.AttnParams()

        if not exists(param_filename):
            print("Starting new model {} with params:".format(args.model))
            params.dump()
            params.save(param_filename)
        else:
            loaded_params.load(param_filename)
            print("Found model also named {} with params:".format(args.model))
            loaded_params.dump()

            # Found pre-existing training with current_epoch
            current_epoch = loaded_params["current_epoch"]

            # Allow for increasing number of epochs of pre-trained model
            if params["epochs"] > loaded_params["epochs"]:
                print(
                    "Number of epochs increased to {} from {} - autoencoder may require further training. If so, property predictor may be trained from scratch.".format(
                        params["epochs"], loaded_params["epochs"]))
                loaded_params["ae_trained"] = False
                loaded_params["epochs"] = params["epochs"]

            elif params["pp_epochs"] > loaded_params["pp_epochs"]:
                print(
                    "Number of property predictor epochs increased to {} from {}, but autoencoder is fully trained. Property predictor will continue training.".format(
                        params["pp_epochs"], loaded_params["pp_epochs"]))
                loaded_params["pp_epochs"] = params["pp_epochs"]

            if loaded_params["ae_trained"]:
                print(
                    "Found model with fully trained auto-encoder.\nProperty predictor has been trained for {} epochs - continuing from epoch {}".format(
                        current_epoch - 1,
                        current_epoch))
            elif current_epoch != 1:
                print(
                    "Found model trained for {} epochs - continuing from epoch {}".format(current_epoch - 1,
                                                                                          current_epoch))

                params = loaded_params

        # Train model
        # if params["pp_weight"] == 0 or params["pp_weight"] is None:
        #     data_train = data_train
        #     data_test = data_test
        # else:
        #     data_train = [data_train, props_train]
        #     data_test = [data_test, props_test]

        model, results = trainTransformer(params=params, data_file=args.data,
                                          # data_train=data_train, data_test=data_test, tokens=tokens,
                                          model_dir=model_dir)


if __name__ == '__main__':
    main()
