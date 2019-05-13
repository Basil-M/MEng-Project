import argparse
from functools import reduce

import h5py
import numpy as np
import pandas
import progressbar
import pickle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from os.path import exists
from dataloader import SmilesToArray
import pygtrie  # For prefix tree

MAX_NUM_ROWS = 1000
SMILES_COL_NAME = 'structure'
DICT_LOC = 'data/CHEMBL_dict.txt'
INFILE = 'data/zinc12.h5'
OUTFILE = 'data/zinc_1k.h5'

from utils import rdkit_funcs, TokenList, rdkit_funcs_mol
from rdkit import Chem


def get_arguments():
    parser = argparse.ArgumentParser(description='Prepare data for training')

    parser.add_argument('--infile', type=str, help='Input file name',
                        default=INFILE)
    parser.add_argument('--outfile', type=str, help='Output file name',
                        default=OUTFILE)
    parser.add_argument('--length', type=int, metavar='N', default=MAX_NUM_ROWS,
                        help='Maximum number of rows to include (randomly sampled).')
    parser.add_argument('--smiles_column', type=str, default=SMILES_COL_NAME,
                        help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    parser.add_argument('--property_column', type=str, default="HBD/HBA/Charge/psf",
                        help="Names of columns (separated by /) containing properties e.g. HOMO9/LUMO9. Default: None")
    parser.add_argument('--rdkit_props', type=str, metavar='prop1/prop2', default='QED/SAS/LOGP/MOLWT',
                        help="Names of properties (separated by /) to fetch from RDKIT e.g. QED/SAS. Options: QED, SAS, LOGP, MOLWT. Default: None")
    parser.add_argument('--model_arch', type=str, metavar='N', default='VAE',
                        help='Model architecture to use - options are VAE, TRANSFORMER')
    parser.add_argument('--train_frac', type=float, metavar='0.8', default=0.8,
                        help='Fraction of data to use for training')
    parser.add_argument('--charset', type=str, metavar='dict.txt', default=DICT_LOC,
                        help='Path of charset file. If blank, will generate charset using data.')
    parser.add_argument('--max_len', type=int, metavar='args.max_len', default=120,
                        help='Maximum length to use from dataset')
    return parser.parse_args()


def main():
    args = get_arguments()
    np.random.seed(1830)
    print("Loading data from", args.infile)
    data = pandas.read_hdf(args.infile, 'table')

    colnames = list(data)

    # We save a list of canonicalised chemicals so that we can later check
    # Whether generated molecules were in the dataset
    pkl_file = args.infile.replace(".h5", ".pkl")
    if not exists(pkl_file):
        print("Generating prefix tree of canonicalised chemicals")
        print("May take a while...")
        num_mols = len(data[args.smiles_column])
        canon_struct_tree = pygtrie.StringTrie()
        for idx in progressbar.progressbar(range(num_mols)):
            mol = Chem.MolFromSmiles(''.join(data[args.smiles_column][idx]))
            if mol:
                canon_name = Chem.MolToSmiles(mol)
                canon_struct_tree[canon_name] = canon_name

        with open(pkl_file, mode='wb') as f:
            pickle.dump(canon_struct_tree, f, pickle.HIGHEST_PROTOCOL)
        del canon_struct_tree
 
    keys = data[args.smiles_column].map(len) < args.max_len + 1
    n_samples = int(1.1*args.length)
    if n_samples <= len(keys):
        # sample an extra 10% to allow for some unwieldy smiles
        data = data[keys].sample(n=int(n_samples))
    else:
        data = data[keys]

    structures = data[args.smiles_column].map(lambda x: list(x.ljust(args.max_len)))

    properties = []
    prop_names = []
    if args.property_column:
        print("Fetching properties from dataset")
        colnames = list(data)
        for prop in args.property_column.split("/"):
            if prop in colnames:
                print("\tFetching", prop)
                properties.append(data[prop][keys])
                prop_names.append(prop)
            else:
                print("\tProperty", prop, "not found in data file.")
                print("\tAvailable columns are:", ','.join(colnames))




    del data

    # Get training and testing indices
    train_idx, test_idx = map(np.array,
                              train_test_split(structures.index, test_size=1 - args.train_frac))

    # Get character set
    if args.charset:
        with open(args.charset) as fin:
            charset = list(ll for ll in fin.read().split('\n') if ll != "")
    else:
        charset = list(reduce(lambda x, y: set(y) | x, structures, set()))
        print(charset)
        return (0,0)
    # Explicitly encode so it can be saved in h5 file
    # PREPROCESSING FOR TRANSFORMER MODEL
    print("Fetching", args.length, "canonicalised strings from dataset")
    bar_i = 0
    widgets = [' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ', ]
    # convert prefix tree to list so it can be indexed by id
    with progressbar.ProgressBar(maxval=args.length, widgets=widgets) as bar:
        train_str = []
        test_str = []
        n_samples = (args.length*np.array([args.train_frac, 1 - args.train_frac])).astype(np.int)
        for (idx, lst, ns) in zip([train_idx, test_idx], [train_str, test_str], n_samples):
            # canonicalise
            for (i, id) in enumerate(idx):
                if len(lst) == ns:
                    break
                mol = ''.join(structures[id])
                rd_mol = Chem.MolFromSmiles(mol)
                if rd_mol:
                    lst.append(Chem.MolToSmiles(rd_mol))
                    bar_i += 1
                else:
                    print("Could not canonicalise molecule", i)
#                    lst.append(mol)

                bar.update(bar_i)
    # Split testing and training data
    # TODO(Basil): Fix ugly hack with the length of Xs...
    h5f = h5py.File(args.outfile, 'w')
    tokens = TokenList(charset)
    transformer_train = SmilesToArray(train_str, tokens, length=args.max_len)
    transformer_test = SmilesToArray(test_str, tokens, length=args.max_len)

    h5f.create_dataset('cat/train', data=transformer_train)
    h5f.create_dataset('cat/test', data=transformer_test)
    h5f.create_dataset('charset', data=[c.encode('utf8') for c in charset])

    # Create data for GRU and pad to max length
    # bs x ml x 38
    onehot_train = to_categorical(transformer_train, tokens.num())
    onehot_test = to_categorical(transformer_test, tokens.num())
    mlen = np.max([args.max_len, np.shape(onehot_train)[1], np.shape(onehot_test)[1]])
    onehot_train = np.pad(onehot_train, ((0, 0), (0, mlen - np.shape(onehot_train)[1]), (0, 0)), 'constant')
    onehot_test = np.pad(onehot_test, ((0, 0), (0, mlen - np.shape(onehot_test)[1]), (0, 0)), 'constant')
    h5f.create_dataset('onehot/train', data=onehot_train)
    h5f.create_dataset('onehot/test', data=onehot_test)

    # PROPERTIES
    test_props = []
    train_props = []

    p = lambda x: np.reshape(np.array(x), newshape=(1, len(x)))
    for prop in properties:
        if len(test_props):
            test_props = np.vstack((test_props, p(prop[test_idx])))
            train_props = np.vstack((train_props, p(prop[train_idx])))
        else:
            test_props = p(prop[test_idx])
            train_props = p(prop[train_idx])

    print("Calculating RDKit properties")
    for prop in args.rdkit_props.split("/"):
        if prop in rdkit_funcs:
            print("\tComputing", prop)

            if len(test_props):
                test_props = np.vstack((test_props, p([rdkit_funcs[prop](s) for s in test_str])))
                train_props = np.vstack((train_props, p([rdkit_funcs[prop](s) for s in train_str])))
            else:
                test_props = p([rdkit_funcs[prop](s) for s in test_str])
                train_props = p([rdkit_funcs[prop](s) for s in train_str])

            prop_names.append(prop)
        else:
            print("\tProperty", prop, "not found as RDKit function")

    print("Normalising properties")
    means = []
    stds = []
    for k in range(len(prop_names)):
        means.append(np.mean(train_props[k, :]))
        stds.append(np.std(train_props[k, :]))
        print(prop_names[k], "- Mu = {:.2f}, Std = {:.2f}".format(means[k],stds[k]))
        train_props[k, :] = (train_props[k, :] - means[k])/stds[k]
        test_props[k, :] = (test_props[k, :] - means[k]) / stds[k]

    if prop_names:
        # save to dataset
        print("TRAIN PROPS SHAPE", np.shape(train_props))
        h5f.create_dataset('properties/names', data=np.string_(prop_names))
        h5f.create_dataset('properties/train', data=np.transpose(train_props))
        h5f.create_dataset('properties/test', data=np.transpose(test_props))
        h5f.create_dataset('properties/means', data=means)
        h5f.create_dataset('properties/stds', data=stds)


if __name__ == '__main__':
    main()
