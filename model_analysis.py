import argparse
## LOGGING
import logging
from os.path import exists

import h5py
import numpy as np
import progressbar
from keras.optimizers import Adam
from scipy.stats import kurtosis

from molecules.model import TriTransformer, MoleculeVAE, SequenceInference
from utils import load_dataset, load_properties, AttnParams

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
from utils import rdkit_funcs
from rdkit import Chem

IBUPROFEN_SMILES = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
ZINC_RANDOM = 'C[NH+](CC1CCCC1)[C@H]2CCC[C@@H](C2)O'

import contextlib
import os, sys


@contextlib.contextmanager
def supress_stderr():
    """
    A context manager to temporarily disable stderr
    """
    stdchannel = sys.stderr
    dest_filename = os.devnull
    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder analysis')
    parser.add_argument('--model_path', type=str, help='Path to model directory e.g. models/VA_192/',
                        default="models/gru/")
    parser.add_argument('--beam_width', type=int, help='Beam width e.g. 5. If 1, will use greedy decoding.',
                        default=5)
    parser.add_argument('--n_seeds', type=int, help='Number of seeds to use latent exploration',
                        default=109, metavar='100')
    parser.add_argument('--n_decodings', type=int, help='Number of decodings for each seed for latent exploration',
                        default=2, metavar='2')
    parser.add_argument('--plot_kd', type=bool, help='Plot distributions of test data in latent space',
                        default=True, metavar='True')
    parser.add_argument('--prior_sample', type=bool, help='If true, will sample from random prior N(0,1)',
                        default=False, metavar='False')
    return parser.parse_args()


def latent_distributions(latents_file, plot_kd=False):
    # n_bins = 100

    with h5py.File(latents_file) as dfile:
        z_test = np.array(dfile['z_test'])
        z_train = np.array(dfile['z_train'])

        k_test = kurtosis(z_test)
        k_train = kurtosis(z_train)
        # print("\tTraining:\n\t\tMean\t{:.2f}\n\t\tStd\t{:.2f}".format(np.mean(k_train), np.std(k_train)))
        print("\tTesting:\n\t\tMean\t{:.2f}\n\t\tStd\t{:.2f}".format(np.mean(k_test), np.std(k_test)))

    if plot_kd:
        # Get relevant modules
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()

        latent_dim = np.shape(z_test)[1]
        from distutils.spawn import find_executable
        if find_executable('latex'):
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            xlab = "$z_i$"
        else:
            xlab = "z_i"

        print("Performing kernel density estimates:")
        for i in progressbar.progressbar(range(latent_dim)):
            sns.distplot(z_test[:, i], hist=False, kde=True,
                         kde_kws={'linewidth': 1})
        plt.xlabel(xlab)
        plt.ylabel("KD")

        filepath = os.path.dirname(latents_file) + "/kd_est.png"
        print("Saving kd plot in ", filepath)
        plt.savefig(filepath)


def property_distributions(data_test, props_test, num_seeds, num_decodings, SeqInfer: SequenceInference, beam_width=1,
                           latents_file=None):
    pst = props_test
    # Get num_seeds random points from test data
    indices = np.array(range(0, len(data_test)))
    np.random.shuffle(indices)
    indices = indices[0:num_seeds]
    data_test = np.array(data_test)  # allow for fancy indexing
    data_test = list(data_test[indices])
    props_test = props_test[indices, :]

    output_molecules = []
    # define decoding function

    if beam_width == 1:
        output = lambda x, y: SeqInfer.decode_sequence_fast(input_seq=seq, moments=[x, y])
    else:
        output = lambda x, y: np.array(SeqInfer.beam_search(input_seq=seq, topk=beam_width, moments=[x, y]))

    # progressbar
    # decode molecules multiple times
    gen_props = []
    bar_i = 0
    widgets = [
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]
    with progressbar.ProgressBar(maxval=num_seeds * num_decodings, widgets=widgets) as bar:
        for seq in data_test:
            # get mean/variance
            mu, logvar = SeqInfer.encode.predict_on_batch(np.expand_dims(seq, 0))
            for dec_itr in range(num_decodings):
                # c_output = output(mu,logvar)
                s = output(mu, logvar)
                if s.ndim > 1: s = s[:, 0]

                with supress_stderr():
                    for mol in s:
                        # keep if unique
                        if mol not in output_molecules:
                            output_molecules.append(mol)
                            mol = Chem.MolFromSmiles(mol)
                            # mol is None if it wasn't a valid SMILES string
                            if mol:
                                try:
                                    gen_props.append([rdkit_funcs[key](mol) for key in rdkit_funcs])
                                except:
                                    print("Could not calculate properties for {}".format(Chem.MolToSmiles(mol)))
                bar_i += 1
                bar.update(bar_i)

    print("Generated {} unique sequences, of which {} were valid.".format(len(output_molecules), len(gen_props)))

    if latents_file:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()

        from distutils.spawn import find_executable

        prop_labels = ["QED", "LogP", "MolWt", "SAS"]
        with h5py.File(latents_file) as dfile:
            z_test = np.array(dfile['z_test'])
            for k in range(len(prop_labels)):
                plt.scatter(z_test[:, 0], z_test[:, 1], c=pst[:, k], s=1)
                plt.xlabel("z_0")
                plt.ylabel("z_1")
                plt.title(prop_labels[k])
                plt.show()

    return np.array(gen_props), props_test, len(gen_props) / len(output_molecules)


def rand_mols(nseeds, latent_dim, SeqInfer: SequenceInference, beam_width=1):
    '''

    :param nseeds:
    :param latent_dim:
    :param SeqInfer:
    :param beam_width:
    :return:
    '''
    output_molecules = []
    # define decoding function

    if beam_width == 1:
        output = lambda x: SeqInfer.decode_sequence_fast(input_seq=None, moments=[x])
    else:
        output = lambda x: np.array(SeqInfer.beam_search(input_seq=None, topk=beam_width, moments=[x]))

    # progressbar
    # decode molecules multiple times
    gen_props = []
    widgets = [
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]

    with progressbar.ProgressBar(maxval=nseeds, widgets=widgets) as bar:
        for bar_i in range(nseeds):
            z_i = np.random.randn(latent_dim)
            s = output(z_i)
            if s.ndim > 1: s = s[:, 0]

            with supress_stderr():
                for mol in s:
                    # keep if unique
                    if mol not in output_molecules:
                        output_molecules.append(mol)
                        print("GENERATED ", mol)
                        mol = Chem.MolFromSmiles(mol)
                        # mol is None if it wasn't a valid SMILES string
                        if mol:
                            try:
                                gen_props.append([rdkit_funcs[key](mol) for key in rdkit_funcs])
                            except:
                                print("Could not calculate properties for {}".format(Chem.MolToSmiles(mol)))
            bar.update(bar_i)

    print("Generated {} unique sequences, of which {} were valid.".format(len(output_molecules), len(gen_props)))

    return np.array(gen_props), len(gen_props) / len(output_molecules)


def main():
    args = get_arguments()
    model_dir = args.model_path

    # Get Params
    model_params = AttnParams()
    model_params.load(model_dir + "params.pkl")
    # Get data
    d_file = model_params["data"]
    data_train, data_test, tokens = load_dataset(d_file, model_params["model_arch"], False)
    props_train, props_test, prop_labels = load_properties(d_file)

    if model_params["model_arch"] == "TRANSFORMER":
        # Model is an attention based model
        model = TriTransformer(tokens, model_params)
        model.build_models()
        model.compile_vae(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
    else:
        # Model is GRU
        model = MoleculeVAE(tokens, model_params)

    SeqInfer = SequenceInference(model, tokens, weights_file=model_dir + "best_model.h5")

    # Assess how close each dimension is to a Gaussian
    # Try to load property training data
    if not exists(model_dir + "latents.h5"):
        print("Generating latent representations from auto-encoder")
        z_train = model.encode_sample.predict([data_train], 64)
        z_test = model.encode_sample.predict([data_test], 64)

        with h5py.File(model_dir + "latents.h5", 'w') as dfile:
            dfile.create_dataset('z_test', data=z_test)
            dfile.create_dataset('z_train', data=z_train)

    print("KURTOSIS:")
    # latent_distributions(model_dir + 'latents.h5', plot_kd=True)

    # Test random molecule
    print("Example decodings with ibruprofen (beam width = 5):")
    print("\tIbuprofen smiles:\t{}".format(IBUPROFEN_SMILES))
    s = SeqInfer.beam_search(IBUPROFEN_SMILES, 1)
    [print("\t\tDecoding {}:\t\t{}".format(i + 1, seq[0])) for (i, seq) in enumerate(s)]

    print("Exploring property distributions of chemicals from {} decoding(s) of {} random seed(s):".format(
        args.n_decodings,
        args.n_seeds))

    if args.prior_sample:
        gen_props, frac_valid = rand_mols(args.n_seeds, model_params["latent_dim"], SeqInfer, args.beam_width)
    else:
        gen_props, data_props, frac_valid = property_distributions(data_test, props_test,
                                                                   num_seeds=args.n_seeds,
                                                                   num_decodings=args.n_decodings,
                                                                   SeqInfer=SeqInfer,
                                                                   beam_width=args.beam_width)  # ,

    print("\tValid mols:\t {:.2f}".format(frac_valid))
    for key in rdkit_funcs:
        if key in prop_labels:
            k = prop_labels.index(key)

            print("\t{}:".format(key))
            dat = props_test[:, k]
            print("\t\tTest distribution:\t {:.2f} ± {:.2f}".format(np.mean(dat), np.std(dat)))

            gen_dat = gen_props[:, k]
            print("\t\tGenerated distribution:\t {:.2f} ± {:.2f}".format(np.mean(gen_dat), np.std(gen_dat)))


def delete_if_exists(filename):
    if exists(filename):
        os.remove(filename)


if __name__ == '__main__':
    main()
