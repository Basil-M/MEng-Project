import h5py
import numpy as np
from molecules.model import TriTransformer
from dataloader import SmilesToArray, AttnParams, MakeSmilesDict, MakeSmilesData
from keras.optimizers import Adam
from scipy.stats import rv_continuous, kurtosis
import argparse
from os.path import exists

## LOGGING
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
# Chemical properties
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Crippen import MolLogP as LogP
from rdkit.Chem.QED import qed as QED
from utils import calculateScore as SAS

IBUPROFEN_SMILES = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
ZINC_RANDOM = 'C[NH+](CC1CCCC1)[C@H]2CCC[C@@H](C2)O'


class rv_gaussian(rv_continuous):
    def _pdf(self, x):
        return np.exp(-x ** 2 / 2.) / np.sqrt(2.0 * np.pi)


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder analysis')
    parser.add_argument('--model_path', type=str, help='Path to model directory e.g. models/VA_192/',
                        default="models/avg_model/")
    parser.add_argument('--beam_width', type=int, help='Beam width e.g. 5. If 1, will use greedy decoding.',
                        default=5)
    parser.add_argument('--n_seeds', type=int, help='Number of seeds to use latent exploration',
                        default=150, metavar='100')
    parser.add_argument('--n_decodings', type=int, help='Number of decodings for each seed for latent exploration',
                        default=2, metavar='5')
    return parser.parse_args()


def latent_distributions(latents_file, plot_kd=False):
    # n_bins = 100

    with h5py.File(latents_file) as dfile:
        z_test = np.array(dfile['z_test'])
        z_train = np.array(dfile['z_train'])

        k_test = kurtosis(z_test)
        k_train = kurtosis(z_train)
        print("\tTraining:\n\t\tMean\t{:.2f}\n\t\tStd\t{:.2f}".format(np.mean(k_train), np.std(k_train)))
        print("\tTesting:\n\t\tMean\t{:.2f}\n\t\tStd\t{:.2f}".format(np.mean(k_test), np.std(k_test)))
    #
    # latent_dim = np.shape(z)[1]
    # KL_div = np.zeros(latent_dim)
    # unit_gauss = rv_gaussian(name='unit_normal')
    # for i in range(latent_dim):
    #     # Get the ith latent dimension
    #     z_i = z[:, i]
    #     mu_i = np.mean(z_i)
    #     std_i = np.std(z_i)
    #     z_i = (z_i - mu_i)/std_i
    #
    #     # Kernel density estimate
    #     hist_i = np.histogram(z_i, 100)
    #     z_rv = rv_histogram(hist_i)
    #     KL_div[i] = z_rv.entropy()/unit_gauss.entropy()
    #
    # return KL_div


def property_distributions(test_data_file, num_seeds, num_decodings, attn_model: TriTransformer, beam_width=1):
    # Get random seeds from test_data
    with h5py.File(test_data_file) as dfile:
        z_test = np.array(dfile['test'])
        props_test = np.array(dfile['test_props'])
        Ps_norms = np.array(dfile['property_norms'])

    # Unnormalise properties
    for k in range(len(Ps_norms)):
        props_test[:, k] *= Ps_norms[k, 1]
        props_test[:, k] += Ps_norms[k, 0]

    # Get num_seeds random points from test data
    indices = np.array(range(0, len(z_test)))
    np.random.shuffle(indices)
    indices = indices[0:num_seeds]
    test_data = np.array(z_test)  # allow for fancy indexing
    test_data = list(test_data[indices])
    props_test = props_test[indices, :]

    output_molecules = []
    # decode molecules multiple times
    for dec_itr in range(num_decodings):
        # decode molecules
        if beam_width == 1:
            output_itr = [attn_model.decode_sequence_fast(seq) for seq in test_data]
        else:
            output_itr = []
            for seq in test_data:
                output_itr += [s[0] for s in attn_model.beam_search(seq, beam_width)]

        # only keep if it's unique
        for mol in output_itr:
            if mol not in output_molecules:
                output_molecules.append(mol)

    # get qed vals of input data
    # qed_vals = [Chem.QED.qed(Chem.MolFromSmiles(molecule)) for molecule in test_data]

    gen_props = []
    for molecule in output_molecules:

        mol = Chem.MolFromSmiles(molecule)
        # mol is None if it wasn't a valid SMILES string
        if mol is not None:
            gen_props.append([QED(mol), LogP(mol), MolWt(mol), SAS(mol)])

    print("Generated {} unique sequences, of which {} were valid.".format(len(output_molecules), len(gen_props)))

    return gen_props, props_test, len(gen_props) / len(output_molecules)


def main():
    args = get_arguments()
    model_dir = args.model_path

    # Load in model
    model_params = AttnParams()
    model_params.load(model_dir + "params.pkl")

    # Get data
    d_file = model_params.get("d_file")
    # data = LoadList(d_file)  # List of SMILES strings

    tokens = MakeSmilesDict(d_file, dict_file='data/SMILES_dict.txt')

    # Prepare model
    model = TriTransformer(tokens, model_params)
    model.compile_vae(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
    model.autoencoder.load_weights(model_dir + "best_model.h5", by_name=True)
    model.encode_model.load_weights(model_dir + "best_model.h5", by_name=True)
    model.decode_model.load_weights(model_dir + "best_model.h5", by_name=True)

    # Assess how close each dimension is to a Gaussian
    # Try to load property training data
    if not exists(model_dir + "latents.h5"):
        print("Generating latent representations from auto-encoder")
        data_train, data_test, props_train, props_test = MakeSmilesData(d_file, tokens=tokens,
                                                                        h5_file=d_file.replace('.txt',
                                                                                               '_data.h5'))
        # z_train = model.encode_model.predict([data_train], 64)
        z_test = model.encode_model.predict([data_test], 64)

        with h5py.File(model_dir + "latents.h5", 'w') as dfile:
            dfile.create_dataset('z_test', data=z_test)
            # dfile.create_dataset('z_train', data=z_train)

    print("KURTOSIS:")
    latent_distributions(model_dir + 'latents.h5')

    # Test random molecule
    print("Example decodings with ibruprofen (beam width = 5):")
    print("\tIbuprofen smiles:\t{}".format(IBUPROFEN_SMILES))

    s = model.beam_search(IBUPROFEN_SMILES, 5)
    [print("\t\tDecoding {}:\t\t{}".format(i + 1, seq[0])) for (i, seq) in enumerate(s)]

    print("Exploring property distributions of chemicals from {} decoding(s) of {} random seed(s):".format(
        args.n_decodings,
        args.n_seeds))
    gen_props, data_props, frac_valid = property_distributions(d_file.replace('.txt', '_data.h5'),
                                                               num_seeds=args.n_seeds,
                                                               num_decodings=args.n_decodings,
                                                               attn_model=model,
                                                               beam_width=args.beam_width)
    prop_labels = ["QED", "LogP", "MolWt", "SAS"]
    print("\tValid mols:\t {}".format(frac_valid))
    for k in range(len(prop_labels)):
        print("\t{}:".format(prop_labels[k]))
        gen_dat = gen_props[k, :]
        dat = data_props[k, :]
        print("\t\tData:\t {:.2f} ± {:.2f}".format(np.mean(gen_dat), np.std(gen_dat)))
        print("\t\tGen:\t {:.2f} ± {:.2f}".format(np.mean(dat), np.std(dat)))




if __name__ == '__main__':
    main()
import h5py
import numpy as np
from molecules.model import TriTransformer
from dataloader import SmilesToArray, AttnParams, MakeSmilesDict, MakeSmilesData
from keras.optimizers import Adam
from scipy.stats import rv_continuous, kurtosis
import argparse
from os.path import exists

## LOGGING
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
# Chemical properties
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Crippen import MolLogP as LogP
from rdkit.Chem.QED import qed as QED
from utils import calculateScore as SAS

IBUPROFEN_SMILES = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
ZINC_RANDOM = 'C[NH+](CC1CCCC1)[C@H]2CCC[C@@H](C2)O'


class rv_gaussian(rv_continuous):
    def _pdf(self, x):
        return np.exp(-x ** 2 / 2.) / np.sqrt(2.0 * np.pi)


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder analysis')
    parser.add_argument('--model_path', type=str, help='Path to model directory e.g. models/VA_192/',
                        default="models/avg_model/")
    parser.add_argument('--beam_width', type=int, help='Beam width e.g. 5. If 1, will use greedy decoding.',
                        default=5)
    parser.add_argument('--n_seeds', type=int, help='Number of seeds to use latent exploration',
                        default=150, metavar='100')
    parser.add_argument('--n_decodings', type=int, help='Number of decodings for each seed for latent exploration',
                        default=2, metavar='5')
    return parser.parse_args()


def latent_distributions(latents_file, plot_kd=False):
    # n_bins = 100

    with h5py.File(latents_file) as dfile:
        z_test = np.array(dfile['z_test'])
        z_train = np.array(dfile['z_train'])

        k_test = kurtosis(z_test)
        k_train = kurtosis(z_train)
        print("\tTraining:\n\t\tMean\t{:.2f}\n\t\tStd\t{:.2f}".format(np.mean(k_train), np.std(k_train)))
        print("\tTesting:\n\t\tMean\t{:.2f}\n\t\tStd\t{:.2f}".format(np.mean(k_test), np.std(k_test)))
    #
    # latent_dim = np.shape(z)[1]
    # KL_div = np.zeros(latent_dim)
    # unit_gauss = rv_gaussian(name='unit_normal')
    # for i in range(latent_dim):
    #     # Get the ith latent dimension
    #     z_i = z[:, i]
    #     mu_i = np.mean(z_i)
    #     std_i = np.std(z_i)
    #     z_i = (z_i - mu_i)/std_i
    #
    #     # Kernel density estimate
    #     hist_i = np.histogram(z_i, 100)
    #     z_rv = rv_histogram(hist_i)
    #     KL_div[i] = z_rv.entropy()/unit_gauss.entropy()
    #
    # return KL_div


def property_distributions(test_data_file, num_seeds, num_decodings, attn_model: TriTransformer, beam_width=1):
    # Get random seeds from test_data
    with h5py.File(test_data_file) as dfile:
        z_test = np.array(dfile['test'])
        props_test = np.array(dfile['test_props'])
        Ps_norms = np.array(dfile['property_norms'])

    # Unnormalise properties
    for k in range(len(Ps_norms)):
        props_test[:, k] *= Ps_norms[k, 1]
        props_test[:, k] += Ps_norms[k, 0]

    # Get num_seeds random points from test data
    indices = np.array(range(0, len(z_test)))
    np.random.shuffle(indices)
    indices = indices[0:num_seeds]
    test_data = np.array(z_test)  # allow for fancy indexing
    test_data = list(test_data[indices])
    props_test = props_test[indices, :]

    output_molecules = []
    # decode molecules multiple times
    for dec_itr in range(num_decodings):
        # decode molecules
        if beam_width == 1:
            output_itr = [attn_model.decode_sequence_fast(seq) for seq in test_data]
        else:
            output_itr = []
            for seq in test_data:
                output_itr += [s[0] for s in attn_model.beam_search(seq, beam_width)]

        # only keep if it's unique
        for mol in output_itr:
            if mol not in output_molecules:
                output_molecules.append(mol)

    # get qed vals of input data
    # qed_vals = [Chem.QED.qed(Chem.MolFromSmiles(molecule)) for molecule in test_data]

    gen_props = []
    for molecule in output_molecules:

        mol = Chem.MolFromSmiles(molecule)
        # mol is None if it wasn't a valid SMILES string
        if mol is not None:
            gen_props.append([QED(mol), LogP(mol), MolWt(mol), SAS(mol)])

    print("Generated {} unique sequences, of which {} were valid.".format(len(output_molecules), len(gen_props)))

    return np.array(gen_props), props_test, len(gen_props) / len(output_molecules)


def main():
    args = get_arguments()
    model_dir = args.model_path

    # Load in model
    model_params = AttnParams()
    model_params.load(model_dir + "params.pkl")

    # Get data
    d_file = model_params.get("d_file")
    # data = LoadList(d_file)  # List of SMILES strings

    tokens = MakeSmilesDict(d_file, dict_file='data/SMILES_dict.txt')

    # Prepare model
    model = TriTransformer(tokens, model_params)
    model.compile_vae(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
    model.autoencoder.load_weights(model_dir + "best_model.h5", by_name=True)
    model.encode_model.load_weights(model_dir + "best_model.h5", by_name=True)
    model.decode_model.load_weights(model_dir + "best_model.h5", by_name=True)

    # Assess how close each dimension is to a Gaussian
    # Try to load property training data
    if not exists(model_dir + "latents.h5"):
        print("Generating latent representations from auto-encoder")
        data_train, data_test, props_train, props_test = MakeSmilesData(d_file, tokens=tokens,
                                                                        h5_file=d_file.replace('.txt',
                                                                                               '_data.h5'))
        # z_train = model.encode_model.predict([data_train], 64)
        z_test = model.encode_model.predict([data_test], 64)

        with h5py.File(model_dir + "latents.h5", 'w') as dfile:
            dfile.create_dataset('z_test', data=z_test)
            # dfile.create_dataset('z_train', data=z_train)

    print("KURTOSIS:")
    latent_distributions(model_dir + 'latents.h5')

    # Test random molecule
    print("Example decodings with ibruprofen (beam width = 5):")
    print("\tIbuprofen smiles:\t{}".format(IBUPROFEN_SMILES))

    s = model.beam_search(IBUPROFEN_SMILES, 5)
    [print("\t\tDecoding {}:\t\t{}".format(i + 1, seq[0])) for (i, seq) in enumerate(s)]

    print("Exploring property distributions of chemicals from {} decoding(s) of {} random seed(s):".format(
        args.n_decodings,
        args.n_seeds))
    gen_props, data_props, frac_valid = property_distributions(d_file.replace('.txt', '_data.h5'),
                                                               num_seeds=args.n_seeds,
                                                               num_decodings=args.n_decodings,
                                                               attn_model=model,
                                                               beam_width=args.beam_width)
    prop_labels = ["QED", "LogP", "MolWt", "SAS"]
    print("\tValid mols:\t {}".format(frac_valid))
    for k in range(len(prop_labels)):
        print("\t{}:".format(prop_labels[k]))
        gen_dat = gen_props[:, k]
        dat = data_props[:, k]
        print("\t\tData:\t {:.2f} ± {:.2f}".format(np.mean(gen_dat), np.std(gen_dat)))
        print("\t\tGen:\t {:.2f} ± {:.2f}".format(np.mean(dat), np.std(dat)))




if __name__ == '__main__':
    main()
