import os

import h5py
import numpy as np
from keras.utils import Sequence

import utils
from utils import TokenList


class SMILESgen:
    def __init__(self, data_filename, batch_size, shuffle=True):
        self.data_file = h5py.File(data_filename, 'r')
        self.train_data = SMILES_data(self.data_file['train'], batch_size, shuffle)
        self.test_data = SMILES_data(self.data_file['test'], batch_size, shuffle)

    def __delete__(self):
        self.data_file.close()


class SMILES_data(Sequence):
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.bs = batch_size
        self.indices = list(range(len(self.data)))
        if shuffle:
            np.random.shuffle(self.indices)
        self.ind = 0
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.data) / self.bs))

    def __getitem__(self, idx):
        self.ind += self.bs
        inds = self.indices[self.ind - self.bs:self.ind]
        inds = np.sort(inds).tolist()
        return np.array(self.data[inds]), None

    def on_epoch_end(self):
        self.ind = 0
        if self.shuffle:
            np.random.shuffle(self.indices)


def MakeSmilesDict(fn=None, min_freq=5, dict_file=None):
    if dict_file is not None and os.path.exists(dict_file):
        print('Loading preprocessed dictionary file {}'.format(dict_file))
        lst = utils.LoadList(dict_file)
    else:
        print("Creating dictionary file {}".format(dict_file))
        data = utils.LoadList(fn)
        wdicts = [{}, {}]
        for ss in data:
            for seq, wd in zip(ss, wdicts):
                for w in seq:
                    wd[w] = wd.get(w, 0) + 1

        wd = utils.FreqDict2List(wd)
        lst = [x for x, y in wd if y >= min_freq]

        # Save dictionary
        if dict_file is not None:
            utils.SaveList(lst, dict_file)

        print('Vocabulary size is {}'.format(len(lst)))

    return TokenList(lst)


#
# def MakeSmilesData(fn=None, tokens=None, h5_file=None, max_len=200, train_frac=0.8):
#     if h5_file is not None and os.path.exists(h5_file):
#         print('Loading data from {}'.format(h5_file))
#
#         with h5py.File(h5_file) as dfile:
#             test_data, train_data = dfile['test'][:], dfile['train'][:]
#             test_pps, train_pps = dfile['test_props'][:], dfile['train_props'][:]
#     else:
#         print("Processing data from {}".format(fn))
#         data = utils.LoadList(fn)
#
#         Xs = []
#         Ps = []
#         Ps_norms = []
#         # Get structures
#         for seq in data:
#             mol = Chem.MolFromSmiles(seq)
#
#             Ps.append([QED(mol), LogP(mol), MolWt(mol), SAS(mol)])
#
#             Xs.append(seq)
#
#         # Normalise properties
#         Ps = np.array(Ps)
#         for k in range(np.shape(Ps)[1]):
#             mu = np.mean(Ps[:, k])
#             std = np.std(Ps[:, k])
#             Ps[:, k] = (Ps[:, k] - mu) / std
#             Ps_norms.append([mu, std])
#
#         # Split testing and training data
#         # TODO(Basil): Fix ugly hack with the length of Xs...
#         split_pos = int(np.floor(train_frac * np.max(np.shape(Xs))))
#
#         train_data = SmilesToArray(Xs[:split_pos], tokens, max_len)
#         train_pps = Ps[:split_pos]
#         test_data = SmilesToArray(Xs[split_pos:], tokens, max_len)
#         test_pps = Ps[split_pos:]
#
#         if h5_file is not None:
#             with h5py.File(h5_file, 'w') as dfile:
#                 dfile.create_dataset('test', data=test_data)
#                 dfile.create_dataset('train', data=train_data)
#                 dfile.create_dataset('test_props', data=test_pps)
#                 dfile.create_dataset('train_props', data=train_pps)
#                 dfile.create_dataset('property_norms', data=Ps_norms)
#
#     return train_data, test_data, train_pps, test_pps
#

def SmilesToArray(xs, tokens, max_len=999):
    '''
    Tokenizes list of smiles strings
    :param xs: List of smiles strings e.g. ['C1ccccc','C1cc=O',...]
    :param tokens: Tokens from MakeSmilesDict
    :param max_len: Maximum length to pad to
    :return: Array of tokenized smiles
    '''
    # if isinstance(xs, str):
    #     xs = [xs]
    # elif isinstance(xs[0], list):
    #     xs = [x[0] for x in xs]

    # find longest string
    longest = np.max([len(x) for x in xs])
    longest = min(longest + 2, max_len)

    X = np.zeros((len(xs), longest), dtype='int32')
    X[:, 0] = tokens.startid()
    for i, x in enumerate(xs):
        # print("Padding {}".format(x))
        x = x[:max_len - 2]

        skipnext = False
        k = 1
        for j, z in enumerate(list(x)):
            if skipnext:
                skipnext = False
            else:
                if x[j:j + 2] in tokens.id2t and j + 1 != len(x):
                    X[i, k] = tokens.id(x[j:j + 2])
                    skipnext = True
                else:
                    X[i, k] = tokens.id(z)
                k = k + 1

        X[i, k] = tokens.endid()

    return X


def separableTest(num_per_class, num_classes, max_len, num_chars_each=15):
    # num_chars_each = min(num_chars_each, np.floor(255/num_classes))
    min_len = np.ceil(0.6 * max_len)
    print("MINIMUM LENGTH: {}".format(min_len))
    # choose characters to use for each class
    asciiInd = list(range(35, 35 + num_chars_each * num_classes))
    np.random.shuffle(asciiInd)
    asciiInd = asciiInd[0:num_classes * num_chars_each]

    # get tokens list
    tokens = []
    for a in asciiInd:
        tokens.append(chr(a))
    tokens = TokenList(tokens)

    classInds = np.reshape(asciiInd, [num_classes, num_chars_each])

    trainStrings = []
    trainClasses = np.array([], dtype=np.int)
    for k in range(num_classes):
        trainClasses = np.append(trainClasses, k * np.ones(num_per_class, dtype=np.int))
        for i in range(num_per_class):
            # randomly choose length
            seq_len = max_len
            # seq_len = np.random.randint(min_len, max_len)
            print("SEQUENCE LENGTH: {}".format(seq_len))
            inds = np.random.randint(0, num_chars_each, seq_len)

            seq = []
            for s in classInds[k, inds]:
                seq.append(chr(s))

            seq = ''.join(seq)
            trainStrings.append(seq)
            # print("Sequence {} of class {}: {}".format(i, k, seq))

    trainData = SmilesToArray(trainStrings, tokens, max_len)

    testStrings = []
    testClasses = np.array([], dtype=np.int)
    num_test = int(np.ceil(0.2 * num_per_class))

    for k in range(num_classes):
        testClasses = np.append(testClasses, k * np.ones(num_test, dtype=np.int))
        for i in range(num_test):
            seq_len = max_len
            # seq_len = np.random.randint(min_len, max_len)
            inds = np.random.randint(0, num_chars_each, seq_len)

            seq = []
            for s in classInds[k, inds]:
                seq.append(chr(s))

            seq = ''.join(seq)
            testStrings.append(seq)
            # print("Sequence {} of class {}: {}".format(i, k, seq))

    testData = SmilesToArray(testStrings, tokens, max_len)

    # shuffle
    inds = list(
        range(0, num_per_class * num_classes))  # np.linspace(0, num_per_class*num_classes-1, num_per_class*num_classes)
    np.random.shuffle(inds)
    trainClasses = trainClasses[inds]
    trainData = trainData[inds]

    inds = list(
        range(0, num_test * num_classes))  # int(np.linspace(0, num_test * num_classes - 1, num_test * num_classes))
    np.random.shuffle(inds)
    testClasses = testClasses[inds]
    testData = testData[inds]

    return tokens, trainClasses, trainData, testClasses, testData


def MarkovSimData(num_per_class, max_len):
    # num_chars_each = min(num_chars_each, np.floor(255/num_classes))
    min_len = np.ceil(max_len / 2)
    tokens = ['A', 'B', 'C', 'D']

    p1 = np.array([[0.9, 0.1, 0.0, 0.0],
                   [0.0, 0.2, 0.8, 0.0],
                   [0.0, 0.25, 0.35, 0.5],
                   [0.15, 0.35, 0.5, 0.0]])

    p2 = np.array([[0.0, 0.0, 0.0, 1.0],
                   [0.0, 0.2, 0.3, 0.6],
                   [1.0, 0.0, 0.0, 0.0],
                   [0.8, 0.1, 0.05, 0.05]])

    trainStrings = []
    trainClasses = np.array([], dtype=np.int)
    testStrings = []
    testClasses = np.array([], dtype=np.int)
    num_test = int(np.ceil(0.2 * num_per_class))
    for k in range(2):
        if k == 1:
            P = np.cumsum(p1, 1)
        else:
            P = np.cumsum(p2, 1)

        testClasses = np.append(testClasses, k * np.ones(num_test, dtype=np.int))
        trainClasses = np.append(trainClasses, k * np.ones(num_per_class, dtype=np.int))
        for i in range(num_per_class):
            # randomly choose first one
            seq = [tokens[np.random.randint(0, len(tokens))]]
            seq_len = np.random.randint(min_len, max_len) + 1
            while len(seq) < seq_len + 1:
                # get corresponding row to sample from
                pi = P[tokens.index(seq[-1]), :]

                symb = np.where(pi > np.random.rand())[0][0]
                seq.append(tokens[symb])
            seq = ''.join(seq)
            trainStrings.append(seq)

        for i in range(num_test):
            # randomly choose first one
            seq = [tokens[np.random.randint(0, len(tokens))]]
            seq_len = np.random.randint(min_len, max_len) + 1
            while len(seq) < seq_len + 1:
                # get corresponding row to sample from
                pi = P[tokens.index(seq[-1]), :]
                symb = np.where(pi > np.random.rand())[0][0]
                seq.append(tokens[symb])
            seq = ''.join(seq)
            testStrings.append(seq)

    tokens = TokenList(tokens)
    trainData = SmilesToArray(trainStrings, tokens, max_len)
    testData = SmilesToArray(testStrings, tokens, max_len)

    # shuffle
    inds = list(range(0, num_per_class * 2))  # np.linspace(0, num_per_class*num_classes-1, num_per_class*num_classes)
    np.random.shuffle(inds)
    trainClasses = trainClasses[inds]
    trainData = trainData[inds]

    inds = list(range(0, num_test * 2))  # int(np.linspace(0, num_test * num_classes - 1, num_test * num_classes))
    np.random.shuffle(inds)
    testClasses = testClasses[inds]
    testData = testData[inds]

    return tokens, trainClasses, trainData, testClasses, testData
