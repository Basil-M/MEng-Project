from __future__ import print_function

import argparse
import os
from os import mkdir
from os.path import exists

import numpy as np
import tensorflow as tf
from keras import backend as K
from shutil import rmtree
import utils
from train import trainTransformer

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

NUM_EPOCHS = 20
BATCH_SIZE = 50
LATENT_DIM = 128
RANDOM_SEED = 1403
DATA = 'data/zinc_100k.h5'
# DATA = 'C:\Code\MEng-Project\data\dummy2.txt'
# DATA = 'data/dummy.txt'

## extra imports to set GPU options
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow import ConfigProto, Session
from keras.backend.tensorflow_backend import set_session
from molecules.model import SequenceInference
from model_analysis import rand_mols, property_distributions, supress_stderr

###################################
# Prevent GPU pre-allocation
config = ConfigProto()
config.gpu_options.allow_growth = True
set_session(Session(config=config))

mnames = {"average1": "AVG1",
          "average2": "AVG2",
          "average3": "AVG3",
          "ar1": "AR1",
          "ar2": "AR2",
          "ar_log": "ARlog",
          "ar_slim": "ARslim",
          "gru_attn": "GRUa",
          "gru": "GRU",
          "conv": "CONV"}


def get_arguments():
    default = utils.AttnParams()

    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--data', type=str, help='The HDF5 file containing preprocessed data.',
                        default=DATA)
    parser.add_argument('--models_dir', type=str,
                        help='Path to folder containing model log directories e.g. model/',
                        default="models/")

    ### TRAINING PARAMETERS
    parser.add_argument('--epochs', type=int, metavar='N', default=20,
                        help='Number of epochs to run during training.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=40,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--bottleneck', type=str, metavar='N', default="gru_attn",
                        help='Choice of bottleneck')
    parser.add_argument('--model_size', type=str, metavar='N', default="medium",
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=60,
                        help='Latent dimension')
    parser.add_argument('--use_WAE', type=bool, metavar='N', default=False,
                        help="Choice to use Normal IMQ WAE with s = 2, weight = 10")
    return parser.parse_args()


def main():
    args = get_arguments()
    np.random.seed(1380)
    model_name = "{}_{}_d{}_{}".format(mnames[args.bottleneck], args.model_size, args.latent_dim,
                                       "WAE" if args.use_WAE else "VAE")
    model_dir = args.models_dir + model_name + "/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    else:
        rmtree(model_dir)
        os.mkdir(model_dir)
    # Get default attention parameters
    params = utils.AttnParams()

    # Standard training params
    params["epochs"] = args.epochs
    params["batch_size"] = args.batch_size
    params["kl_pretrain_epochs"] = 2
    params["kl_anneal_epochs"] = 3
    params["bottleneck"] = args.bottleneck
    params["stddev"] = 1
    params["model_arch"] = "TRANSFORMER"
    params["latent_dim"] = args.latent_dim
    # Get training and test data from data file
    # Set up model
    if args.model_size == "small":
        # AVG1:     47741
        # AVG2:     53216
        # GRU:      51076
        # GRU_ATTN: 50806
        # CONV:     51629
        # AR_SLIM:  58394

        params["d_model"] = 32
        params["d_inner_hid"] = 196
        params["d_k"] = 6
        params["heads"] = 6
        params["layers"] = 2
        if params["bottleneck"] == "ar_slim":
            params["ID_layers"] = 3
            params["ID_d_model"] = 6
            params["ID_width"] = 6
            params["ID_d_inner_hid"] = 30
            params["ID_d_k"] = 4
            params["ID_d_v"] = 4
            params["ID_heads"] = 5
        elif "ar" in params["bottleneck"]:
            params["ID_layers"] = 2
            params["ID_d_model"] = 8
            params["ID_d_inner_hid"] = 64
            params["ID_width"] = 4
            params["ID_d_k"] = 6
            params["ID_d_v"] = 6
            params["ID_heads"] = 4
        elif params["bottleneck"] == "gru":
            params["ID_layers"] = 3
            params["ID_d_model"] = 48
        elif params["bottleneck"] == "gru_attn":
            params["ID_layers"] = 3
            params["ID_d_model"] = 42
        elif params["bottleneck"] == "conv":
            params["ID_layers"] = 2     # num layers
            params["ID_d_k"] = 5        # min_filt_size/num
            params["ID_d_model"] = 64   # dense dim

    elif args.model_size == "medium":
        # AVG1:     171413
        # AVG2:     174488
        # GRU:      172664
        # GRU_ATTN: 171308
        # CONV:     171800
        # AR_SLIM:  184968
        params["d_model"] = 64
        params["d_inner_hid"] = 256
        params["d_k"] = 8
        params["heads"] = 8
        params["layers"] = 3

        if params["bottleneck"] == "ar_slim":
            params["ID_layers"] = 4
            params["ID_d_model"] = 8
            params["ID_width"] = 6
            params["ID_d_inner_hid"] = 64
            params["ID_d_k"] = 4
            params["ID_d_v"] = 4
            params["ID_heads"] = 4
        elif "ar" in params["bottleneck"]:
            params["ID_layers"] = 2
            params["ID_d_model"] = 32
            params["ID_width"] = 4
            params["ID_d_inner_hid"] = 196
            params["ID_d_k"] = 7
            params["ID_d_v"] = 7
            params["ID_heads"] = 5
        elif params["bottleneck"] == "gru_attn":
            params["ID_layers"] = 4
            params["ID_d_model"] = 78
        elif params["bottleneck"] == "gru":
            params["ID_layers"] = 4
            params["ID_d_model"] = 82
        elif params["bottleneck"] == "conv":
            params["ID_layers"] = 4
            params["ID_d_k"] = 8
            params["ID_d_model"] = 156

    elif args.model_size == "big":
        # big avg:      1,131,745
        # big ar_log:   1,316,449
        # big GRU:      1,029,152
        # big CONV:     439,419
        params["d_model"] = 128
        params["d_inner_hid"] = 768
        params["d_k"] = 12
        params["heads"] = 12
        params["layers"] = 4

        if "ar" in params["bottleneck"]:
            params["ID_layers"] = 3
            params["ID_d_model"] = 40
            params["ID_width"] = 4
            params["ID_d_inner_hid"] = 256
            params["ID_d_k"] = 8
            params["ID_d_v"] = 8
            params["ID_heads"] = 6
        elif "gru" in params["bottleneck"]:
            params["ID_layers"] = 5
            params["ID_d_model"] = 160
        elif params["bottleneck"] == "conv":
            params["ID_layers"] = 6
            params["ID_d_k"] = 10
            params["ID_d_model"] = 512

    params["d_v"] = params["d_k"]

    if args.use_WAE:
        params["WAE_kernel"] = "IMQ_normal"
        params["kl_max_weight"] = 10
        params["WAE_s"] = 2

    # Handle interim decoder parameters
    params.setIDparams()

    # Create model tracking folder
    if not exists(model_dir):
        mkdir(model_dir)

    # Handle parameters
    param_filename = model_dir + "params.pkl"
    loaded_params = utils.AttnParams()

    if not exists(param_filename):
        print("Starting new model {} with params:".format(model_name))
        params.dump()
        params.save(param_filename)
    else:
        loaded_params.load(param_filename)
        print("Found model also named {} trained for {} epochs with params:".format(model_name,
                                                                                    loaded_params["current_epoch"]))
        loaded_params.dump()
        # Allow for increasing number of epochs of pre-trained model
        if params["epochs"] > loaded_params["epochs"]:
            print(
                "Number of epochs increased to {} from {}. Autoencoder will be trained more.".format(
                    params["epochs"], loaded_params["epochs"]))
            loaded_params["epochs"] = params["epochs"]

        params = loaded_params

    model, results = trainTransformer(params=params, data_file=args.data,
                                      model_dir=model_dir)

    data_train, data_test, _, _, tokens = utils.load_dataset(args.data, "onehot" if args.bottleneck == "conv" else "cat",
                                                             params["pp_weight"])
    props_train, props_test, prop_labels = utils.load_properties(args.data)

    SeqInfer = SequenceInference(model, tokens)

    with supress_stderr():
        seed_output = property_distributions(data_test[0], props_test,
                                             num_seeds=300,
                                             num_decodings=3,
                                             SeqInfer=SeqInfer,
                                             beam_width=5, data_file='data/zinc12.h5')

        rand_output = rand_mols(400, params["latent_dim"], SeqInfer, 5, data_file='data/zinc12.h5')

    if results:
        print("\tValidation accuracy:\t {:.2f}".format(np.amax(results.history['val_acc'])))
        # TODO(Basil): Add getting results from CSV file...

    for (mode, output) in zip(["SAMPLING PRIOR", "SAMPLING WITH SEEDS"], [seed_output, rand_output]):
        print("BY", mode)
        print("\tGenerated {} molecules, of which {} were valid and {} were novel.".format(output["num_mols"], output["num_valid"], output["num_novel"]))
        print("\t\tValid mols:\t {:.2f}".format(output["num_valid"] / output["num_mols"]))
        if "num_novel" in output: print("\t\tNovel mols:\t{:.2f}".format(output["num_novel"]/output["num_valid"]))
        print("\t\tSuccess frac:\t{:.2f}".format(output["success_frac"]))
        print("\t\tYield:\t{:.2f}".format(output["yield"]))
        for (i, key) in enumerate(utils.rdkit_funcs):
            if key in prop_labels:
                k = prop_labels.index(key)

                print("\t\t{}:".format(key))
                dat = props_test[:, k]
                print("\t\t\tTest distribution:\t {:.2f} ± {:.2f}".format(np.mean(dat), np.std(dat)))

                gen_dat = output["gen_props"][:, i]
                print("\t\t\tGenerated distribution:\t {:.2f} ± {:.2f}".format(np.mean(gen_dat), np.std(gen_dat)))


if __name__ == '__main__':
    main()
