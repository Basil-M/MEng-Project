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
from utils import rdkit_funcs, TokenList, AttnParams, rdkit_funcs_mol

# GuacaMol Data Links
urls = {
    "train": "https://ndownloader.figshare.com/files/13612760",
    "test": "https://ndownloader.figshare.com/files/13612757",
    "val": "https://ndownloader.figshare.com/files/13612766",
}

# Path to charset
dict = 'data/guac_dict.txt'
d_file = 'guac.h5'
train_len = 300000
test_len = 25000
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
    if not exists(folder + '/' + d_file):
        print("Could not find prepared h5 file of guac data. Preprocessing.")
        canon = lambda x: x
        canon = lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x))
        print("Loading and canonicalizing guacaMolecules")
        with open(folder + '/guac_test.smiles') as fin:
            test_str = [ll for ll in fin.read().split('\n') if ll != ""]
        with open(folder + '/guac_train.smiles') as fin:
            train_str = [ll for ll in fin.read().split('\n') if ll != ""]

        # keep random data
        np.random.shuffle(test_str)
        np.random.shuffle(train_str)
        test_str = test_str[0:test_len]
        train_str = train_str[0:train_len]
        test_mol = [Chem.MolFromSmiles(x) for x in test_str]
        train_mol = [Chem.MolFromSmiles(x) for x in train_str]
        # Split testing and training data
        print("Creating categorical SMILES arrays")
        h5f = h5py.File(folder + d_file, 'w')
        tokens = TokenList(charset)
        transformer_train = SmilesToArray(train_str, tokens, length=110)
        transformer_test = SmilesToArray(test_str, tokens, length=110)

        h5f.create_dataset('cat/train', data=transformer_train)
        h5f.create_dataset('cat/test', data=transformer_test)
        h5f.create_dataset('charset', data=[c.encode('utf8') for c in charset])

        # Create data for GRU and pad to max length
        # bs x ml x 38
        print("Creating onehot arrays")
#        onehot_train = to_categorical(transformer_train, tokens.num())
#        onehot_test = to_categorical(transformer_test, tokens.num())
#        mlen = np.max([110, np.shape(onehot_train)[1], np.shape(onehot_test)[1]])
#        onehot_train = np.pad(onehot_train, ((0, 0), (0, mlen - np.shape(onehot_train)[1]), (0, 0)), 'constant')
#        onehot_test = np.pad(onehot_test, ((0, 0), (0, mlen - np.shape(onehot_test)[1]), (0, 0)), 'constant')
#        h5f.create_dataset('onehot/train', data=onehot_train)
#        h5f.create_dataset('onehot/test', data=onehot_test)

        # PROPERTIES
        test_props = []
        train_props = []

        p = lambda x: np.reshape(np.array(x), newshape=(1, len(x)))
        print("Calculating RDKit properties")
        prop_names = []
        for prop in rdkit_funcs:
            print("\tComputing", prop)
            if len(test_props):
                test_props = np.vstack((test_props, p([rdkit_funcs_mol[prop](s) for s in test_mol])))
                train_props = np.vstack((train_props, p([rdkit_funcs_mol[prop](s) for s in train_mol])))
            else:
                test_props = p([rdkit_funcs_mol[prop](s) for s in test_mol])
                train_props = p([rdkit_funcs_mol[prop](s) for s in train_mol])

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
    params = defaultParams('big','conv_attn',8,'KQV',latent_dim=96)
#    params["decoder"] = "TRANSFORMER_NoFE"
#    params["model"] += "_NoFE"
    params["epochs"] = 30
    # huge model
    params["AM_softmax"] = False
    model_fol = 'guac/GUAC-' + params['model'] + '/'
    if not exists(model_fol):
        mkdir(model_fol)
    else:
        print("Loading model.")
        params.load(model_fol + "params.pkl")
    params['model'] = model_fol

    print("model for guac:", model_fol)
    model, results = trainTransformer(params=params, data_file='data/'+d_file,
                                      model_dir=model_fol)

    bw = 5
    Gen = TransformerDMG(model, bw, 5)
    assess_distribution_learning(Gen, 'data/guac_train.smiles',
                                 model_fol + '/results_bw{}_npz{}.json'.format(bw, 5))
    Gen = TransformerDMG(model, bw, 1)
    assess_distribution_learning(Gen, 'data/guac_train.smiles',
                                 model_fol + '/results_bw{}_npz{}.json'.format(bw, 1))

def RandomSampler():
    train_file = 'data/guac_train.smiles'

class RandomDMG(DistributionMatchingGenerator):
    def __init__(self, data, tokens):
        self.data = data
        self.tokens = tokens
        self.get_chem = lambda k: ''.join([tokens.id2t[s] for s in self.data[k] if s >=4])
        self.dlen = len(data)

    def generate(self, number_samples: int):
        ind = np.array(range(self.dlen))
        np.random.shuffle(ind)
        inds = np.random.randint(self.dlen,size=(number_samples,))
        return [self.get_chem(i) for i in inds]


class TransformerDMG(DistributionMatchingGenerator):
    def __init__(self, model, beam_width=5, num_per_beam=5):
        self.model = model
        self.bw = beam_width
        self.num_per_beam = num_per_beam
        self.ldim = model.p["latent_dim"]

    def generate(self, number_samples: int):
        print("Generating", number_samples, "samples.")
        return ['C'] * number_samples

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
