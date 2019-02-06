from __future__ import print_function

import argparse
import os
import numpy as np

from keras.optimizers import Adam
from molecules.transformer import LRSchedulerPerStep
from os.path import exists
from os import mkdir, remove
import tensorflow as tf
from keras import backend as k


from utils import epoch_track, WeightAnnealer_epoch, load_dataset

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))
import dataloader as dd

NUM_EPOCHS = 20
BATCH_SIZE = 50
LATENT_DIM = 128
RANDOM_SEED = 1403
DATA = 'data/zinc_100k.txt'
# DATA = 'C:\Code\MEng-Project\data\dummy2.txt'
# DATA = 'data/dummy.txt'
MODEL_ARCH = 'TRANSFORMER'
MODEL_NAME = 'id3'
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
    default = dd.AttnParams()

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
    parser.add_argument('--kl_pretrain_epochs', type=float, metavar='1', default=default.get("kl_pretrain_epochs"),
                        help='Number of epochs to train before introducing KL loss')
    parser.add_argument('--kl_anneal_epochs', type=float, metavar='5', default=default.get("kl_anneal_epochs"),
                        help='Number of epochs to anneal over')
    parser.add_argument('--kl_max_weight', type=float, metavar='1', default=default.get("kl_max_weight"),
                        help='Maximum KL weight')
    parser.add_argument('--RBF_s', type=float, metavar='1', default=default.get("RBF_s"),
                        help='Scale factor to use for RBF for Wasserstein fitting. If 0, will use ELBO instead (default).')

    parser.add_argument('--pp_weight', type=float, metavar='1.5', default=default.get("pp_weight"),
                        help='For joint optimisation: Amount to weight MSE loss of property predictor')

    parser.add_argument('--gen', help='Whether to use generator for data', default=False)

    ### BOTTLENECK PARAMETERS
    parser.add_argument('--latent_dim', type=int, metavar='N', default=default.get("latent_dim"),
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--bottleneck', type=str, default=default.get("bottleneck"),
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
    parser.add_argument('--ID_width', type=int, metavar='N', default=default.get("ID_width"),
                        help='Number of interim decoder layers')

    ### MODEL PARAMETERS
    parser.add_argument('--d_model', type=int, metavar='N', default=default.get("d_model"),
                        help='Dimensionality of transformer model')
    parser.add_argument('--d_inner_hid', type=int, metavar='N', default=default.get("d_inner_hid"),
                        help='Dimensionality of fully connected networks after attention layers')
    parser.add_argument('--d_k', type=int, metavar='N', default=default.get("d_k"),
                        help='Dimensionality of attention keys')
    parser.add_argument('--d_v', type=int, metavar='N', default=default.get("d_v"),
                        help='Dimensionality of attention values')
    parser.add_argument('--heads', type=int, metavar='N', default=default.get("heads"),
                        help='Number of attention heads to use')
    parser.add_argument('--layers', type=int, metavar='N', default=default.get("layers"),
                        help='Number of encoder/decoder layers')

    parser.add_argument('--dropout', type=float, metavar='0.1', default=default.get("dropout"),
                        help='Dropout to use in autoencoder')
    parser.add_argument('--stddev', type=float, metavar='0.01', default=default.get("stddev"),
                        help='Standard deviation of variational sampler')

    ### PROPERTY PREDICTOR PARAMETERS
    parser.add_argument('--pp_epochs', type=int, metavar='N', default=default.get("pp_epochs"),
                        help='Number of epochs to train property predictor')

    parser.add_argument('--pp_layers', type=int, metavar='N', default=default.get("pp_layers"),
                        help='Number of dense layers for property predictor')

    return parser.parse_args()


def main():
    args = get_arguments()
    np.random.seed(args.random_seed)

    MODEL_DIR = args.models_dir + args.model + "/"

    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

    ## VARIATIONAL AUTOENCODER
    if args.model_arch == "VAE":

        from molecules.model import MoleculeVAE as model_arch
        data_train, data_test, charset = load_dataset(args.data)

        model = model_arch()
        if os.path.isfile(args.model):
            model.load(charset, args.model, latent_rep_size=args.latent_dim)
        else:
            model.create(charset, latent_rep_size=args.latent_dim)

        checkpointer = ModelCheckpoint(filepath=MODEL_DIR,
                                       verbose=1,
                                       save_best_only=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      patience=3,
                                      min_lr=args.base_lr / 10)

        model.autoencoder.fit(
            data_train,
            data_train,
            shuffle=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[checkpointer, reduce_lr],
            validation_data=(data_test, data_test)
        )
    else:

        # GET NUMBER OF AVAILABLE GPUS
        CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES')
        if CUDA_VISIBLE_DEVICES is None:
            N_GPUS = 1
        else:
            N_GPUS = len(CUDA_VISIBLE_DEVICES.split(","))

        print("Found {} GPUs".format(N_GPUS))
        print("Making smiles dict")

        # Get default attention parameters
        params = dd.AttnParams()

        # Process data
        params.set("d_file", args.data)
        d_file = args.data
        # tokens = dd.MakeSmilesDict(d_file, dict_file=d_file.replace('.txt', '_dict.txt'))
        tokens = dd.MakeSmilesDict(d_file, dict_file='data/SMILES_dict.txt')

        # Get training and test data from data file
        if args.gen:
            # Use a generator for data
            _, _, props_train, props_test = dd.MakeSmilesData(d_file, tokens=tokens,
                                                              h5_file=d_file.replace('.txt', '_data.h5'))
            gen = dd.SMILESgen(d_file.replace('.txt', '_data.h5'), args.batch_size)
        else:
            data_train, data_test, props_train, props_test = dd.MakeSmilesData(d_file, tokens=tokens,
                                                                               h5_file=d_file.replace('.txt',
                                                                                                      '_data.h5'))
        # Set up model
        for arg in vars(args):
            if arg in params.params:
                params.set(arg, getattr(args, arg))

        # Handle interim decoder parameters
        params.setIDparams()

        # Create model tracking folder
        if not exists(MODEL_DIR):
            mkdir(MODEL_DIR)

        # Handle parameters
        current_epoch = 1
        param_filename = MODEL_DIR + "params.pkl"
        loaded_params = dd.AttnParams()

        if not exists(param_filename):
            print("Starting new model {} with params:".format(args.model))
            params.dump()
            params.save(param_filename)
        else:
            loaded_params.load(param_filename)
            print("Found model also named {} with params:".format(args.model))
            loaded_params.dump()

            # Found pre-existing training with current_epoch
            current_epoch = loaded_params.get("current_epoch")

            # Allow for increasing number of epochs of pre-trained model
            if params.get("epochs") > loaded_params.get("epochs"):
                print(
                    "Number of epochs increased to {} from {} - autoencoder may require further training. If so, property predictor may be trained from scratch.".format(
                        params.get("epochs"), loaded_params.get("epochs")))
                loaded_params.set("ae_trained", False)
                loaded_params.set("epochs", params.get("epochs"))

            elif params.get("pp_epochs") > loaded_params.get("pp_epochs"):
                print(
                    "Number of property predictor epochs increased to {} from {}, but autoencoder is fully trained. Property predictor will continue training.".format(
                        params.get("pp_epochs"), loaded_params.get("pp_epochs")))
                loaded_params.set("pp_epochs", params.get("pp_epochs"))

            if loaded_params.get("ae_trained"):
                print(
                    "Found model with fully trained auto-encoder.\nProperty predictor has been trained for {} epochs - continuing from epoch {}".format(
                        current_epoch - 1,
                        current_epoch))
            elif current_epoch != 1:
                print(
                    "Found model trained for {} epochs - continuing from epoch {}".format(current_epoch - 1,
                                                                                          current_epoch))

                params = loaded_params

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

        # Learning rate scheduler
        callbacks = []
        callbacks.append(LRSchedulerPerStep(params.get("d_model"),
                                            int(
                                                4 / args.base_lr)))  # there is a warning that it is slow, however, it's ok.


        # Model saver
        callbacks.append(ModelCheckpoint(MODEL_DIR + "model.h5", save_best_only=False,
                                         save_weights_only=True))

        # Best model saver
        callbacks.append(ModelCheckpoint(MODEL_DIR + "best_model.h5", save_best_only=True,
                                         save_weights_only=True))

        # Tensorboard Callback
        callbacks.append(TensorBoard(log_dir=MODEL_DIR + "logdir/",
                                     histogram_freq=0,
                                     batch_size=args.batch_size,
                                     write_graph=True,
                                     write_images=True,
                                     update_freq='batch'))

        if args.bottleneck == "none":
            model.compile_vae(Adam(args.base_lr, 0.9, 0.98), N_GPUS=N_GPUS)
        else:
            # avoid exploding gradients
            model.compile_vae(Adam(args.base_lr, 0.9, 0.98, epsilon=1e-9, clipnorm=1.0, clipvalue=0.5), N_GPUS=N_GPUS)

        try:
            model.autoencoder.load_weights(MODEL_DIR + "model.h5")
        except:
            print("New model.")
            params.save(param_filename)

        # Set up epoch tracking callback
        callbacks.append(epoch_track(params, param_filename=param_filename))
        current_epoch = params.get("current_epoch")
        # Weight annealer callback
        if params.get("stddev") != 0 and params.get("stddev") is not None:
            callbacks.append(WeightAnnealer_epoch(model.kl_loss_var,
                                                  anneal_epochs=params.get("kl_anneal_epochs"),
                                                  max_val=params.get("kl_max_weight"),
                                                  init_epochs=params.get("kl_pretrain_epochs")))

        # Train model
        try:
            if not params.get("ae_trained"):
                print("Training autoencoder.")

                # Delete any existing datafiles containing latent representations
                if exists(MODEL_DIR + "latents.h5"):
                    remove(MODEL_DIR + "latents.h5")

                if args.gen and params.get("pp_weight") == 0.0:
                    print("Using generator for data!")
                    model.autoencoder.fit_generator(gen.train_data, None,
                                                    epochs=args.epochs, initial_epoch=current_epoch - 1,
                                                    validation_data=gen.test_data,
                                                    callbacks=callbacks)
                else:
                    if params.get("pp_weight") == 0.0:
                        train = data_train
                        test = data_test
                    else:
                        train = [data_train, props_train]
                        test = [data_test, props_test]
                    print("Not using generator for data.")
                    model.autoencoder.fit(train, None, batch_size=args.batch_size*N_GPUS,
                                          epochs=args.epochs, initial_epoch=current_epoch - 1,
                                          validation_data=(test, None),
                                          callbacks=callbacks)

        except KeyboardInterrupt:
            # print("Interrupted on epoch {}".format(ep_track.epoch()))
            print("Training interrupted.")


if __name__ == '__main__':
    main()
