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

MAX_NUM_ROWS = 100000
SMILES_COL_NAME = 'structure'
DICT_LOC = 'data/SMILES_dict.txt'
INFILE = 'data/zinc12.h5'
OUTFILE = 'data/zinc_100k.h5'

from utils import rdkit_funcs, TokenList
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

    print("Loading data from", args.infile)
    data = pandas.read_hdf(args.infile, 'table')

    colnames = list(data)

    # We save a list of canonicalised chemicals so that we can later check
    # Whether generated molecules were in the dataset
    txt_file = args.infile.replace(".h5", ".pkl")
    if not exists(txt_file):
        print("Generating text file of canonicalised chemicals")
        print("May take a while...")
        num_mols = len(data[args.smiles_column])
        canon_structs = []
        for idx in progressbar.progressbar(range(num_mols)):
            mol = Chem.MolFromSmiles(''.join(data[args.smiles_column][idx]))
            if mol:
                canon_structs.append(Chem.MolToSmiles(mol))
        with open(txt_file, mode='wb') as f:
            pickle.dump(canon_structs, f, pickle.HIGHEST_PROTOCOL)
    else:
        canon_structs = pickle.load(txt_file)

    keys = data[args.smiles_column].map(len) < args.max_len + 1
    if args.length <= len(keys):
        data = data[keys].sample(n=args.length)
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
    # Explicitly encode so it can be saved in h5 file
    # PREPROCESSING FOR TRANSFORMER MODEL
    print("Canonicalising strings")
    bar_i = 0
    widgets = [' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ', ]
    with progressbar.ProgressBar(maxval=args.length, widgets=widgets) as bar:
        train_str = []
        test_str = []
        for (idx, lst) in zip([train_idx, test_idx], [train_str, test_str]):
            # canonicalise
            for id in idx:
                lst.append(canon_structs[id])
                bar_i += 1
                bar.update(bar_i)
    # Split testing and training data
    # TODO(Basil): Fix ugly hack with the length of Xs...
    h5f = h5py.File(args.outfile, 'w')
    tokens = TokenList(charset)
    transformer_train = SmilesToArray(train_str, tokens, args.max_len)
    transformer_test = SmilesToArray(test_str, tokens, args.max_len)

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
