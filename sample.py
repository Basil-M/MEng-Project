from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys

from molecules.model import MoleculeVAE
from utils import one_hot_array, one_hot_index, from_one_hot_array, decode_smiles_from_indexes, load_dataset, TokenList, AttnParams

from pylab import figure, axes, scatter, title, show

from rdkit import Chem
from rdkit.Chem import Draw

LATENT_DIM = 292
TARGET = 'autoencode'
DATA = "data/zinc_5k.h5"
MODEL = "models/gru/"
def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencode network')
    parser.add_argument('--data', type=str, help='File of latent representation tensors for decoding.', default=DATA)
    parser.add_argument('--model', type=str, help='Trained Keras model to use.', default=MODEL)
    parser.add_argument('--save_h5', type=str, help='Name of a file to write HDF5 output to.')
    parser.add_argument('--target', type=str, default=TARGET,
                        help='What model to sample from: autoencode, encode, decode.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()

def read_latent_data(filename):
    h5f = h5py.File(filename, 'r')
    data = h5f['latent_vectors'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    return (data, charset)

def autoencoder(args, model):
    latent_dim = args.latent_dim
    data, charset = load_dataset(args.data, split = False)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    sampled = model.autoencode.predict(data[0].reshape(1, 120, len(charset))).argmax(axis=2)[0]
    mol = decode_smiles_from_indexes(map(from_one_hot_array, data[0]), charset)
    sampled = decode_smiles_from_indexes(sampled, charset)
    print(mol)
    print(sampled)

def decode(args, model):
    latent_dim = args.latent_dim
    data, charset = read_latent_data(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    sampled = model.decode.predict(data[0].reshape(1, latent_dim)).argmax(axis=2)[0]
    sampled = decode_smiles_from_indexes(sampled, charset)
    print(sampled)

def encode(args, model):
    latent_dim = args.latent_dim
    data, charset = load_dataset(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    x_latent = model.encode.predict(data)
    if args.save_h5:
        h5f = h5py.File(args.save_h5, 'w')
        h5f.create_dataset('charset', data = charset)
        h5f.create_dataset('latent_vectors', data = x_latent)
        h5f.close()
    else:
        np.savetxt(sys.stdout, x_latent, delimiter = '\t')

def main():
    args = get_arguments()
    data_train, data_test, tokens = load_dataset(args.data, "VAE")
    params = AttnParams()
    params.load(args.model + "params.pkl")
    model = MoleculeVAE(tokens, params)

    IBUPROFEN_SMILES = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
    sampled = model.encode_sample.predict(tokens.onehotify(IBUPROFEN_SMILES, params["len_limit"]))
    print(sampled, np.shape(sampled))
    sampled = model.decode.predict(sampled.reshape(1, params["latent_dim"]))
    print(sampled, np.shape(sampled))
    sampled = sampled.argmax(axis=2)[0]
    print(sampled, np.shape(sampled))
    print("Final output", ''.join([tokens.id2t[s] for s in sampled]))
    # if args.target == 'autoencoder':
    #     autoencoder(args, model)
    # elif args.target == 'encode':
    #     encode(args, model)
    # elif args.target == 'decode':
    #     decode(args, model)

if __name__ == '__main__':
    main()
