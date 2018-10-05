from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import Callback
from molecules.transformer import LRSchedulerPerStep, LRSchedulerPerEpoch
from os.path import exists
from os import mkdir

import dataloader as dd

NUM_EPOCHS = 30
BATCH_SIZE = 128
LATENT_DIM = 292
RANDOM_SEED = 1337
DATA = 'data/processed.txt'
MODEL_ARCH = 'ATTN'
MODEL_NAME = 'attn'
MODEL_DIR = 'models/'

# MODEL = 'models/model.h5'
# MODEL_LOGDIR


class epoch_track(Callback):
    def __init__(self, params, param_filename):
        self._params = params
        self._filename = param_filename

    def epoch(self):
        return self._epoch

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self._params.set("current_epoch", self._params.get("current_epoch")+1)
        self._params.save(self._filename)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--data', type=str, help='The HDF5 file containing preprocessed data.',
                        default=DATA)
    parser.add_argument('--model', type=str,
                        help='Name of the model - e.g. attn_dh128. The folder model_dir/model_name/ will contain saved models, parameters and tensorboard log-directories',
                        default=MODEL_NAME)
    parser.add_argument('--models_dir', type=str,
                        help='Path to folder containing model log directories e.g. model/',
                        default=MODEL_DIR)
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
                        help='Seed to use to start randomizer for shuffling.')
    parser.add_argument('--model_arch', type=str, metavar='N', default=MODEL_ARCH,
                        help='Model architecture to use - options are VAE and ATTN')

    return parser.parse_args()


def main():
    args = get_arguments()
    np.random.seed(args.random_seed)

    if args.model_arch == 'VAE':
        from molecules.model import MoleculeVAE as model_arch
    else:
        from molecules.model import Transformer as model_arch

    from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
        decode_smiles_from_indexes, load_dataset
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

    if args.model_arch == "VAE":
        data_train, data_test, charset = load_dataset(args.data)
        print("charset")
        model = model_arch()
        if os.path.isfile(args.model):
            model.load(charset, args.model, latent_rep_size=args.latent_dim)
        else:
            model.create(charset, latent_rep_size=args.latent_dim)

        checkpointer = ModelCheckpoint(filepath=args.model,
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
            nb_epoch=args.epochs,
            batch_size=args.batch_size,
            callbacks=[checkpointer, reduce_lr],
            validation_data=(data_test, data_test)
        )
    else:
        d_model = 512
        d_file = args.data

        print("Making smiles dict")
        # Process data
        tokens = dd.MakeSmilesDict(d_file, dict_file=d_file.replace('.txt', '_dict.txt'))
        data_train, data_test = dd.MakeSmilesData(d_file, tokens=tokens,
                                                  h5_file=d_file.replace('.txt', '_data.h5'))

        # Set up model
        params = dd.attn_params()
        params.set("d_h",args.latent_dim)
        if not exists(args.models_dir + args.model + "/"):
            mkdir(args.models_dir + args.model + "/")
        model = model_arch(tokens, params)

        lr_scheduler = LRSchedulerPerStep(d_model, 4000)  # there is a warning that it is slow, however, it's ok.

        model_saver = ModelCheckpoint(args.models_dir + args.model + "/model.h5", save_best_only=True, save_weights_only=True)
        tbCallback = TensorBoard(log_dir=args.models_dir + args.model + "/logdir/",
                                 histogram_freq=0,
                                 batch_size=args.batch_size,
                                 write_graph=True,
                                 write_images=True,
                                 update_freq='batch')



        model.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
        current_epoch = 1
        param_filename = args.models_dir+args.model+"/params.pkl"
        try:
            loaded_params = dd.attn_params()
            if not exists(param_filename):
                print("New model")
                params.save(param_filename)
            else:
                loaded_params.load(param_filename)

                if params.equals(loaded_params) == 0:
                    print("Found model also named {} but with different parameters:".format(args.model))
                    loaded_params.dump()
                    print("Please choose a different model name.")
                    return 0
                else:
                    # Found pre-existing training with current_epoch
                    current_epoch = loaded_params.get("current_epoch")
                    if current_epoch != 1:
                        print("Found model trained for {} epochs - continuing from epoch {}".format(current_epoch-1, current_epoch))
                        params=loaded_params
                model.model.load_weights(args.models_dir + args.model + "/model.h5")
        except:
            print("New model")
            params.save(param_filename)


        # Set up epoch tracking callback
        ep_track = epoch_track(params, param_filename=param_filename)

        # Train model
        try:
            model.model.fit([data_train, data_train], None, batch_size=args.batch_size,
                            epochs=args.epochs, initial_epoch=current_epoch - 1,
                            validation_data=([data_test, data_test], None),
                            callbacks=[lr_scheduler, model_saver, tbCallback, ep_track])
        except KeyboardInterrupt:
            print("Interrupted on epoch {}".format(ep_track.epoch()))

if __name__ == '__main__':
    main()
