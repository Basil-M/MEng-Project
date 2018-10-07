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
BATCH_SIZE = 64
LATENT_DIM = 292
RANDOM_SEED = 1337
DATA = 'data/zinc_10k.txt'
MODEL_ARCH = 'ATTN'
MODEL_NAME = 'attn'
MODEL_DIR = 'models/'
MODEL_NAME = 'vae_attn_10k'
# MODEL = 'models/model.h5'
# MODEL_LOGDIR


class epoch_track(Callback):
    def __init__(self, params, param_filename):
        self._params = params
        self._filename = param_filename

    def on_epoch_end(self, epoch, logs={}):
        self._params.set("current_epoch", self._params.get("current_epoch") + 1)
        self._params.save(self._filename)
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
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[checkpointer, reduce_lr],
            validation_data=(data_test, data_test)
        )
    else:
        # new = False
        # if new:
        # data_train_vae, data_test_vae, charset_vae = load_dataset('data/zinc_10k.h5')

        # d_file = args.data
        # tokens = dd.MakeSmilesDict(d_file, dict_file=d_file.replace('.txt', '_dict.txt'))
        # data_train, data_test = dd.MakeSmilesData(d_file, tokens=tokens,
        #                                           h5_file=d_file.replace('.txt', '_data.h5'))
        # charset = tokens.id2t
        #
        # model = model_arch()
        #
        # # Create model tracking folder
        # if not exists(args.models_dir + args.model + "/"):
        #     mkdir(args.models_dir + args.model + "/")
        #
        # # Set up model
        # params = dd.attn_params()
        # params.set("d_h", args.latent_dim)
        # params.set("batch_size", args.batch_size)
        #
        # # Resuming from previous training
        #
        # current_epoch = 1
        # param_filename = args.models_dir + args.model + "/params.pkl"
        # try:
        #     loaded_params = dd.attn_params()
        #     if not exists(param_filename):
        #         print("New model")
        #         params.save(param_filename)
        #     else:
        #         loaded_params.load(param_filename)
        #
        #         if params.equals(loaded_params) == 0:
        #             print("Found model also named {} but with different parameters:".format(args.model))
        #             loaded_params.dump()
        #             print("Please choose a different model name.")
        #             return 0
        #         else:
        #             # Found pre-existing training with current_epoch
        #             current_epoch = loaded_params.get("current_epoch")
        #             if current_epoch != 1:
        #                 print("Found model trained for {} epochs - continuing from epoch {}".format(current_epoch - 1,
        #                                                                                             current_epoch))
        #                 params = loaded_params
        #
        #         model.load(tokens,
        #                    weights_file=args.models_dir + args.model + "/model.h5",
        #                    params_file=param_filename)
        # except:
        #     print("New model")
        #     params.save(param_filename)
        #     model.create(tokens, params)
        #
        # lr_scheduler = LRSchedulerPerStep(params.get("d_model"), 4000)  # there is a warning that it is slow, however, it's ok.
        #
        # model_saver = ModelCheckpoint(args.models_dir + args.model + "/model.h5", save_best_only=True, save_weights_only=True)
        # tbCallback = TensorBoard(log_dir=args.models_dir + args.model + "/logdir/",
        #                          histogram_freq=0,
        #                          batch_size=args.batch_size,
        #                          write_graph=True,
        #                          write_images=True,
        #                          update_freq='batch')
        # # Set up epoch tracking callback
        # ep_track = epoch_track(params, param_filename=param_filename)
        #
        #
        # model.autoencoder.fit(
        #     data_train, None,
        #     shuffle=True,
        #     initial_epoch=current_epoch-1,
        #     epochs=args.epochs,
        #     batch_size=args.batch_size,
        #     callbacks=[lr_scheduler, model_saver, tbCallback, ep_track],
        #     validation_data=(data_test, None)
        # )

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

        # Create model tracking folder
        if not exists(args.models_dir + args.model + "/"):
            mkdir(args.models_dir + args.model + "/")

        # Set up model
        model = model_arch(tokens, params)

        lr_scheduler = LRSchedulerPerStep(d_model, 4000)  # there is a warning that it is slow, however, it's ok.

        model_saver = ModelCheckpoint(args.models_dir + args.model + "/model.h5", save_best_only=True, save_weights_only=True)
        tbCallback = TensorBoard(log_dir=args.models_dir + args.model + "/logdir/",
                                 histogram_freq=0,
                                 batch_size=args.batch_size,
                                 write_graph=True,
                                 write_images=True,
                                 update_freq='batch')
        model.compile_vae(Adam(0.001, 0.9, 0.98, epsilon=1e-9))

        # Resuming from previous training
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
                model.autoencoder.load_weights(args.models_dir + args.model + "/model.h5")
        except:
            print("New model")
            params.save(param_filename)


        # Set up epoch tracking callback
        ep_track = epoch_track(params, param_filename=param_filename)

        # Train model
        try:
            model.autoencoder.fit([data_train, data_train], None, batch_size=args.batch_size,
                            epochs=args.epochs, initial_epoch=current_epoch - 1,
                            validation_data=([data_test, data_test], None),
                            callbacks=[lr_scheduler, model_saver, tbCallback, ep_track])
        except KeyboardInterrupt:
            print("Interrupted on epoch {}".format(ep_track.params.get("current_epoch")))


if __name__ == '__main__':
    main()
