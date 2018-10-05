import os, sys, time, random

from typing import Dict, Any, Union

import ljqpy
import h5py
import numpy as np
import time
import csv
import pickle

class attn_params:
    _params = None

    def __init__(self):
        self._params = {
            "current_epoch": 1,
            "epochs":20,
            "batch_size":64,
            "len_limit":120,
            "d_model": 512,
            "d_inner_hid": 512,
            "n_head": 8,
            "d_k": 64,
            "d_v": 64,
            "layers": 3,
            "dropout": 0.1,
            "d_h": 128,
            "share_word_emb": True
        }

    def get(self, param):
        if param in self._params:
            return self._params[param]
        else:
            raise Warning("Param {} unrecognised".format(param))

    def set(self, param, value):
        if not value is None:
            if param in self._params:
                self._params[param] = value
            else:
                print("Param {} unrecognised".format(param))

    def load(self, fn):
        with open(fn, mode= 'rb') as f:
            self._params = pickle.load(f)


    def save(self, fn):
        with open(fn, mode='wb') as f:
            pickle.dump(self._params, f, pickle.HIGHEST_PROTOCOL)

    def dump(self):
        # get max length
        m_len = max([len(key) for key in self._params])

        for key, value in self._params.iteritems():
            print("\t{}  {}".format(key.ljust(m_len), value))

    def equals(self, other_params):
        for key, value  in self._params.iteritems():
            if other_params._params[key] != value and key != "current_epoch":
                return False
        return True

    @property
    def params(self):
        return self._params


class TokenList:
    def __init__(self, token_list):
        self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
        self.t2id = {v: k for k, v in enumerate(self.id2t)}

    def id(self, x):    return self.t2id.get(x, 1)

    def token(self, x):    return self.id2t[x]

    def num(self):        return len(self.id2t)

    def startid(self):  return 2

    def endid(self):    return 3


def MakeSmilesDict(fn=None, min_freq=5, dict_file=None):
    if dict_file is not None and os.path.exists(dict_file):
        print('Loading preprocessed dictionary file {}'.format(dict_file))
        lst = ljqpy.LoadList(dict_file)
    else:
        print("Creating dictionary file {}".format(dict_file))

        data = ljqpy.LoadCSV(fn)
        wdicts = [{}, {}]
        for ss in data:
            for seq, wd in zip(ss, wdicts):
                for w in seq:
                    wd[w] = wd.get(w, 0) + 1
        wd = ljqpy.FreqDict2List(wd)
        lst = [x for x, y in wd if y >= min_freq]

        # Save dictionary
        if dict_file is not None:
            ljqpy.SaveList(lst, dict_file)

        print('Vocabulary size is {}'.format(len(lst)))

    return TokenList(lst)

def MakeSmilesData(fn=None, tokens=None, h5_file=None, max_len=200, train_frac=0.8):
    if h5_file is not None and os.path.exists(h5_file):
        print('Loading data from {}'.format(h5_file))

        with h5py.File(h5_file) as dfile:
            test_data, train_data = dfile['test'][:], dfile['train'][:]
    else:
        print("Processing data from {}".format(fn))
        data = ljqpy.LoadCSVg(fn)

        Xs = []

        # Get structures
        for seq in data:
            Xs.append(list(seq))

        # Randomise
        random.shuffle(Xs)

        # Split testing and training data
        # TODO(Basil): Fix ugly hack with the length of Xs...
        split_pos = int(np.floor(train_frac*np.max(np.shape(Xs))))
        print("Xs shape: {} \nSplit pos: {}".format(np.shape(Xs), split_pos))

        train_data = pad_smiles(Xs[:split_pos], tokens, max_len)

        test_data = pad_smiles(Xs[split_pos:], tokens, max_len)

        print("Train_data shape: {}\nTest_data shape:{}".format(np.shape(train_data), np.shape(test_data)))
        #
        # for d in train_data:
        #     print(d)

        if h5_file is not None:
            with h5py.File(h5_file, 'w') as dfile:
                dfile.create_dataset('test', data=test_data)
                dfile.create_dataset('train', data=train_data)
    return train_data, test_data

def pad_smiles(xs, tokens, max_len=999):
    longest = np.max([len(x[0]) for x in xs])
    longest = min(longest+2, max_len)

    # print("Longest: {}".format(longest))
    X = np.zeros((len(xs), longest), dtype='int32')
    X[:, 0] = tokens.startid()
    for i, x in enumerate(xs):
        # print("Padding {}".format(x))
        x = x[0]
        x = x[:max_len - 2]
        for j, z in enumerate(list(x)):
            # print("Handling character {}".format(z))
            X[i, 1 + j] = tokens.id(z)
        X[i, 1 + len(x)] = tokens.endid()
    return X

