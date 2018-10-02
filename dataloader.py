import os, sys, time, random
import ljqpy
import h5py
import numpy as np
import time

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

