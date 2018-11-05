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
import tensorflow as tf
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
k.tensorflow_backend.set_session(tf.Session(config=config))
import dataloader as dd

NUM_EPOCHS = 50
BATCH_SIZE = 10
LATENT_DIM = 128
RANDOM_SEED = 1337
DATA = 'data/zinc_1k.txt'
MODEL_ARCH = 'ATTN_ID'
MODEL_NAME = 'attn'
MODEL_NAME = 'LT2'
MODEL_DIR = 'models/'

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
                        help='Model architecture to use - options are VAE, ATTN and ATTN_ID')

    parser.add_argument('--gen', help='Whether to use generator for data')
    ### MODEL PARAMETERS
    parser.add_argument('--latent_dim', type=int, metavar='N', default=default.get("latent_dim"),
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--d_model', type=int, metavar='N', default=default.get("d_model"),
                        help='Dimensionality of transformer model')
    parser.add_argument('--d_inner_hid', type=int, metavar='N', default=default.get("d_inner_hid"),
                        help='Dimensionality of fully connected networks after attention layers')
    parser.add_argument('--d_k', type=int, metavar='N', default=default.get("d_k"),
                        help='Dimensionality of attention keys')
    parser.add_argument('--d_v', type=int, metavar='N', default=default.get("d_v"),
                        help='Dimensionality of attention values')
    parser.add_argument('--n_head', type=int, metavar='N', default=default.get("n_head"),
                        help='Number of attention heads to use')
    parser.add_argument('--layers', type=int, metavar='N', default=default.get("layers"),
                        help='Number of encoder/decoder layers')
    parser.add_argument('--dropout', type=float, metavar='0.1', default=default.get("dropout"),
                        help='Dropout to use in autoencoder')
    parser.add_argument('--epsilon', type=float, metavar='0.01', default=default.get("epsilon"),
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

    if args.model_arch == 'VAE':
        from molecules.model import MoleculeVAE as model_arch
    else:
        from molecules.model import TriTransformer as model_arch

    from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
        decode_smiles_from_indexes, load_dataset
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

    MODEL_DIR = args.models_dir + args.model + "/"

    if args.model_arch == "VAE":
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
                                      min_lr=0.0001)

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
        print("Making smiles dict")

        params = dd.AttnParams()

        # Process data
        params.set("d_file", args.data)
        d_file = args.data
        tokens = dd.MakeSmilesDict(d_file, dict_file=d_file.replace('.txt', '_dict.txt'))
        if args.gen:
            _, _, props_train, props_test = dd.MakeSmilesData(d_file, tokens=tokens,
                                                              h5_file=d_file.replace('.txt', '_data.h5'))
            gen = dd.SMILESgen(d_file.replace('.txt', '_data.h5'), args.batch_size)
        else:
            data_train, data_test, props_train, props_test = dd.MakeSmilesData(d_file, tokens=tokens,
                                                                               h5_file=d_file.replace('.txt', '_data.h5'))



        # Set up model
        for arg in vars(args):
            if arg in params.params:
                params.set(arg, getattr(args, arg))

        # Create model tracking folder
        if not exists(MODEL_DIR):
            mkdir(MODEL_DIR)

        # Handle parameters
        current_epoch = 1
        param_filename = MODEL_DIR + "params.pkl"
        loaded_params = dd.AttnParams()
        if not exists(param_filename):
            print("New model - params didn't exist.")
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

        if params.get("model_arch") == "ATTN":
            from molecules.model import Transformer as model_arch
        else:
            from molecules.model import TriTransformer as model_arch

        # Set up model
        model = model_arch(tokens, params)

        # Learning rate scheduler
        lr_scheduler = LRSchedulerPerStep(params.get("d_model"),
                                          4000)  # there is a warning that it is slow, however, it's ok.

        # Model saver
        model_saver = ModelCheckpoint(MODEL_DIR + "model.h5", save_best_only=False,
                                      save_weights_only=True)

        best_model_saver = ModelCheckpoint(MODEL_DIR + "best_model.h5", save_best_only=True,
                                           save_weights_only=True)

        # Tensorboard Callback
        tbCallback = TensorBoard(log_dir=MODEL_DIR + "logdir/",
                                 histogram_freq=0,
                                 batch_size=args.batch_size,
                                 write_graph=True,
                                 write_images=True,
                                 update_freq='batch')

        model.compile_vae(Adam(0.001, 0.9, 0.98, epsilon=1e-9, clipnorm=1.0, clipvalue=0.5))
        model.autoencoder.summary()
        try:
                model.autoencoder.load_weights(MODEL_DIR + "model.h5")
        except:
            print("New model.")
            params.save(param_filename)

        # Set up epoch tracking callback
        ep_track = epoch_track(params, param_filename=param_filename)

        # Train model
        try:
            if not params.get("ae_trained"):
                print("Training autoencoder.")

                # Delete any existing datafiles containing latent representations
                if exists(MODEL_DIR + "latents.h5"):
                    remove(MODEL_DIR + "latents.h5")

                if args.gen:
                    print("Using generator for data!")
                    model.autoencoder.fit_generator(gen.train_data, None,
                                                    epochs=args.epochs, initial_epoch=current_epoch-1,
                                                    validation_data=gen.test_data,
                                                    callbacks=[lr_scheduler, model_saver, best_model_saver, tbCallback,
                                                               ep_track])
                else:
                    print("Not using generator for data.")
                    model.autoencoder.fit(data_train, None, batch_size=args.batch_size,
                                          epochs=args.epochs, initial_epoch=current_epoch - 1,
                                          validation_data=(data_test, None),
                                          callbacks=[lr_scheduler, model_saver, best_model_saver, tbCallback, ep_track])

            print("Autoencoder training complete. Loading best model.")
            model.autoencoder.load_weights(MODEL_DIR + "best_model.h5")

            # Try to load property training data
            if not exists(MODEL_DIR + "latents.h5"):
                print("Generating latent representations from auto-encoder for property predictor training.")
                z_train = model.output_latent.predict([data_train, data_train], 64)
                z_test = model.output_latent.predict([data_test, data_test], 64)


                s = z_train[0]

                with h5py.File(MODEL_DIR + "latents.h5", 'w') as dfile:
                    dfile.create_dataset('z_test', data=z_test)
                    dfile.create_dataset('z_train', data=z_train)

            else:
                print("Loading previously generated latent representations for property predictor training.")
                with h5py.File(MODEL_DIR + "latents.h5") as dfile:
                    z_test, z_train = dfile['z_test'][:], dfile['z_train'][:]

            # Strange hack for dimensionality
            props_test = np.expand_dims(props_test, 3)
            props_train = np.expand_dims(props_train, 3)
            model.property_predictor.compile(optimizer='adam',
                                             loss='mean_squared_error')
            # Model saver
            model_saver = ModelCheckpoint(MODEL_DIR + "pp_model.h5", save_best_only=False,
                                          save_weights_only=True)

            best_model_saver = ModelCheckpoint(MODEL_DIR + "best_pp_model.h5", save_best_only=True,
                                               save_weights_only=True)

            # Load in property predictor weights
            try:
                model.property_predictor.load_weights(MODEL_DIR + "pp_model.h5", by_name=True)
            except:
                pass

            print("Training property predictor")
            model.property_predictor.fit(z_train, np.expand_dims(props_train, 3),
                                         batch_size=params.get("batch_size"), epochs=params.get("pp_epochs"),
                                         initial_epoch=params.get("current_epoch") - 1,
                                         validation_data=(z_test, np.expand_dims(props_test, 3)),
                                         callbacks=[model_saver, best_model_saver, tbCallback, ep_track])

            try:
                model.property_predictor.load_weights(MODEL_DIR + "best_pp_model.h5", by_name=True)
            except:
                pass

        except KeyboardInterrupt:
            print("Interrupted on epoch {}".format(ep_track.epoch()))


if __name__ == '__main__':
    main()
