import tensorflow as tf
from os.path import exists
from os import mkdir
from urllib.request import urlretrieve
from default_models import defaultParams
import h5py
import numpy as np
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from keras.utils import to_categorical
from rdkit import Chem

from dataloader import SmilesToArray
from train import trainTransformer
from utils import rdkit_funcs, TokenList, AttnParams

# GuacaMol Data Links
urls = {
    "train": "https://ndownloader.figshare.com/files/13612760",
    "test": "https://ndownloader.figshare.com/files/13612757",
    "val": "https://ndownloader.figshare.com/files/13612766",
}

# Path to charset
dict = 'data/guac_dict.txt'


def preprocess_guac_data(folder):
    # Get character set
    with open(dict) as fin:
        charset = list(ll for ll in fin.read().split('\n') if ll != "")

    if not exists(folder + "/guac_test.smiles"):
        print("Could not find guac data files. Downloading.")
        for key in urls:
            urlretrieve(urls[key], folder + "/guac_" + key + ".smiles")
    # Explicitly encode so it can be saved in h5 file
    # Preprocessing
    if not exists(folder + '/guac.h5'):
        print("Could not find prepared h5 file of guac data. Preprocessing.")
        canon = lambda x: x
        #canon = lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x))
        print("Loading and canonicalizing guacaMolecules")
        with open(folder + '/guac_test.smiles') as fin:
            test_str = [ll for ll in fin.read().split('\n') if ll != ""]
        with open(folder + '/guac_train.smiles') as fin:
            train_str = [ll for ll in fin.read().split('\n') if ll != ""]

        # Split testing and training data
        print("Creating categorical SMILES arrays")
        h5f = h5py.File(folder + "guac.h5", 'w')
        tokens = TokenList(charset)
        transformer_train = SmilesToArray(train_str, tokens, length=110)
        transformer_test = SmilesToArray(test_str, tokens, length=110)

        h5f.create_dataset('cat/train', data=transformer_train)
        h5f.create_dataset('cat/test', data=transformer_test)
        h5f.create_dataset('charset', data=[c.encode('utf8') for c in charset])

        # Create data for GRU and pad to max length
        # bs x ml x 38
        print("Creating onehot arrays")
        onehot_train = to_categorical(transformer_train, tokens.num())
        onehot_test = to_categorical(transformer_test, tokens.num())
        mlen = np.max([110, np.shape(onehot_train)[1], np.shape(onehot_test)[1]])
        onehot_train = np.pad(onehot_train, ((0, 0), (0, mlen - np.shape(onehot_train)[1]), (0, 0)), 'constant')
        onehot_test = np.pad(onehot_test, ((0, 0), (0, mlen - np.shape(onehot_test)[1]), (0, 0)), 'constant')
        h5f.create_dataset('onehot/train', data=onehot_train)
        h5f.create_dataset('onehot/test', data=onehot_test)

        # PROPERTIES
        test_props = []
        train_props = []

        p = lambda x: np.reshape(np.array(x), newshape=(1, len(x)))
        print("Calculating RDKit properties")
        prop_names = []
        for prop in rdkit_funcs:
            print("\tComputing", prop)
            if len(test_props):
                test_props = np.vstack((test_props, p([rdkit_funcs[prop](s) for s in test_str])))
                train_props = np.vstack((train_props, p([rdkit_funcs[prop](s) for s in train_str])))
            else:
                test_props = p([rdkit_funcs[prop](s) for s in test_str])
                train_props = p([rdkit_funcs[prop](s) for s in train_str])

            prop_names.append(prop)

        print("Normalising properties")
        means = []
        stds = []
        for k in range(len(prop_names)):
            means.append(np.mean(train_props[k, :]))
            stds.append(np.std(train_props[k, :]))
            print(prop_names[k], "- Mu = {:.2f}, Std = {:.2f}".format(means[k], stds[k]))
            train_props[k, :] = (train_props[k, :] - means[k]) / stds[k]
            test_props[k, :] = (test_props[k, :] - means[k]) / stds[k]

        if prop_names:
            # save to dataset
            h5f.create_dataset('properties/names', data=np.string_(prop_names))
            h5f.create_dataset('properties/train', data=np.transpose(train_props))
            h5f.create_dataset('properties/test', data=np.transpose(test_props))
            h5f.create_dataset('properties/means', data=means)
            h5f.create_dataset('properties/stds', data=stds)


def main():
    # preprocess
    preprocess_guac_data('data/')

    # define model
    params = defaultParams('medium','avg',8,'KQV',latent_dim=96)
    model_fol = 'models/GUAC-' + params['model']
    if not exists(model_fol):
        mkdir(model_fol)
    params['model'] = model_fol

    model, results = trainTransformer(params=params, data_file='data/guac.h5',
                                      model_dir=model_fol)

    Gen = TransformerDMG(model, 5, 5)
    assess_distribution_learning(Gen, 'data/guac_train.smiles',
                                 model_fol + '/results_bw{}_npz{}.json'.format(5, 5))


class TransformerDMG(DistributionMatchingGenerator):
    def __init__(self, model, beam_width=5, num_per_beam=5):
        self.model = model
        self.bw = beam_width
        self.num_per_beam = num_per_beam
        self.ldim = model.p["latent_dim"]

    def generate(self, number_samples: int):
        out = []
        while len(out) < number_samples:
            z_i = np.random.randn(self.ldim)
            out_i = []
            s = self.model.decode_from_sample(z_i, beam_width=self.bw)
            s = np.array(s)
            if s.ndim > 1: s = s[:, 0]
            for mol in s:
                if mol not in out and mol not in out_i:
                    out_i.append(mol)
                    # Keep a maximum of num_per_beam for each
                    # latent point
                    if len(out_i) == self.num_per_beam:
                        break
            out.extend(out_i)
        return out[:number_samples]

if __name__ == '__main__':
    main()
