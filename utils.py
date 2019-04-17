# coding = utf-8
import os
import pickle
from os.path import dirname
import time
import h5py
import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from keras.utils import to_categorical
from rdkit.Chem.Crippen import MolLogP as LogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.QED import qed as QED


def FreqDict2List(dt):
    return sorted(dt.items(), key=lambda d: d[-1], reverse=True)


def LoadList(fn):
    with open(fn) as fin:
        st = list(ll for ll in fin.read().split('\n') if ll != "")
    return st


def LoadDict(fn, func=str):
    dict = {}
    with open(fn) as fin:
        for lv in (ll.split('\t', 1) for ll in fin.read().split('\n') if ll != ""):
            dict[lv[0]] = func(lv[1])
    return dict


class epoch_track(Callback):
    '''
    Callback which tracks the current epoch
    And updates the pickled Params file in the model directory
    '''

    def __init__(self, params, param_filename, csv_track=False):
        super().__init__()
        self._params = params
        self._filename = param_filename
        if csv_track:
            models_dir = dirname(dirname(param_filename))
            self.csv_filename = models_dir + "/runs.csv"
            self.rownum, _ = params.dumpToCSV(self.csv_filename)
        else:
            self.csv_filename = None

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_time = time.time() - self.epoch_time
        self._params["epoch_time"] = (self._params["epoch_time"]*epoch + self.epoch_time)/(epoch+1)
        self._params["current_epoch"] += 1
        self._params.save(self._filename)
        epoch += 1
        if self.csv_filename is not None:
            # load csv
            arr = np.genfromtxt(self.csv_filename, delimiter=",", dtype=str)
            # pad column if necessary
            num_params = len(self._params.params)
            if np.shape(arr)[1] < num_params + 2 * epoch:
                padding = np.zeros((np.shape(arr)[0], num_params + 2 * epoch - np.shape(arr)[1]))

                arr = np.hstack((arr, padding))
                for e in range(epoch):
                    arr[0, num_params + 2 * e] = "Epoch {} train acc".format(e + 1)
                    arr[0, num_params + 2 * e + 1] = "Epoch {} val acc".format(e + 1)

            # add new val/training acc
            arr[self.rownum, num_params + 2 * epoch - 2] = logs["acc"]
            arr[self.rownum, num_params + 2 * epoch - 1] = logs["val_acc"]
            # save csv
            np.savetxt(self.csv_filename, arr, delimiter=",", fmt='%s')
        self.epoch_time = time.time()
        return

    def on_train_end(self, logs={}):
        self._params.save(self._filename)
        return

    def epoch(self):
        return self._params["current_epoch"]


class WeightAnnealer_epoch(Callback):
    '''Weight of variational autoencoder scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            weight for the VAE (float).
        Currently just adjust kl weight, will keep xent weight constant
    '''

    def __init__(self, var, anneal_epochs=10, max_val=1.0, init_epochs=1):
        super().__init__()
        self.weight_var = var
        self.max_weight = max_val
        self.init_epochs = init_epochs
        self.epochs_to_max = anneal_epochs
        print(
            "Annealing KL weight to a max value of {};\n\tFirst training {} epochs without KL loss\n\tThen annealing using tanh schedule over {} epochs".format(
                max_val, init_epochs, anneal_epochs))
        # print("Initialising weight loss annealer with w0 = {} and increment of {} each epoch".format(start_val, b))
        # print("\tExpected to reach max_val of {} in {} epochs".format(max_val, int(
        #     self.init_epochs + np.ceil(np.log(max_val / start_val) / np.log(1.5)))))
        # print("\tFirst training {} epochs without variational term".format(self.init_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        if self.weight_var is not None:
            # weight = min(self.max_weight, self.w0 * (self.inc ** (epoch - self.init_epochs)))
            epoch = epoch - self.init_epochs
            if epoch < 0:
                weight = 0
            elif epoch >= self.epochs_to_max:
                weight = self.max_weight
            else:
                weight = 0.5 * self.max_weight * (1 + np.tanh(-2.5 + 5 * epoch / self.epochs_to_max))

            print("Current KL loss annealer weight is {}".format(weight))
            K.set_value(self.weight_var, weight)

    #
    #
    # def __init__(self, schedule, weight, weight_orig, weight_name):
    #     super(WeightAnnealer_epoch, self).__init__()
    #
    #     # self.schedule = schedule
    #     self.weight_var = weight
    #     self.weight_orig = weight_orig
    #     self.weight_name = weight_name
    #
    # def on_epoch_begin(self, epoch, logs=None):
    #     if logs is None:
    #         logs = {}
    #     new_weight = self.schedule(epoch)
    #     new_value = new_weight * self.weight_orig
    #     print("Current {} annealer weight is {}".format(self.weight_name, new_value))
    #     assert type(
    #         new_weight) == float, 'The output of the "schedule" function should be float.'
    #     K.set_value(self.weight_var, new_value)


# UTILS FOR RECURSIVE VAE
def one_hot_array(i, n):
    return map(int, [ix == i for ix in range(n)])


def one_hot_index(vec, charset):
    return map(charset.index, vec)


def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0,):
        return None
    return int(oh[0][0])


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()


def load_dataset(filename, data_format="cat", props=False):
    with h5py.File(filename, 'r') as h5f:
        data_train = h5f[data_format + '/train'][:]
        data_test = h5f[data_format + '/test'][:]
        tokens = TokenList([s.decode('utf-8') for s in h5f['charset'][:]])
        if props:
            props_train = h5f['properties/train'][:]
            props_test = h5f['properties/test'][:]
            # data_train = [data_train, props_train]
            # data_test = [data_test, props_test]
            return data_train, data_test, props_train, props_test, tokens
        else:
            return data_train, data_test, None, None, tokens


def load_properties(filename):
    with h5py.File(filename, 'r') as h5f:
        props_train = h5f['properties/train'][:]
        props_test = h5f['properties/test'][:]
        prop_labels = h5f['properties/names'][:]
        means = h5f['properties/means'][:]
        stds = h5f['properties/stds'][:]

    for k in range(len(prop_labels)):
        props_train[:,k]*=stds[k]
        props_train[:,k]+=means[k]
        props_test[:,k]*=stds[k]
        props_test[:,k]+=means[k]

    prop_labels = [prop.decode('utf-8') for prop in prop_labels]
    return props_train, props_test, prop_labels
# SA SCORE
#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
# from rdkit.six.moves import cPickle
from rdkit.six import iteritems
import pickle as cPickle
import math

import os.path as op

_fscores = None


def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)  # + "/"
    _fscores = cPickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

def str2mol(s):
    if isinstance(s, str):
        return Chem.MolFromSmiles(s)
    else:
        return s

rdkit_funcs = {"QED": lambda x: QED(str2mol(x)),
               "MOLWT": lambda x: MolWt(str2mol(x)),
               "SAS": lambda x: calculateScore(str2mol(x)),
               "LOGP": lambda x: LogP(str2mol(x))}


class AttnParams:
    _params = None

    def __init__(self):
        self._params = {
            "model": None,
            "data": None,  # Data stuff
            "len_limit": 120,
            "num_props": 4,
            "current_epoch": 1,  # Training params
            "epochs": 20,
            "batch_size": 40,
            "kl_pretrain_epochs": 1,
            "kl_anneal_epochs": 1,
            "kl_max_weight": 1,
            "WAE_kernel": None,
            "WAE_s": 2,
            "stddev": 1,
            "pp_weight": 1.25,
            "decoder": "TRANSFORMER",  # Model params
            "latent_dim": 32,
            "d_model": 24,
            "d_inner_hid": 196,
            "d_k": 4,
            "d_v": 4,
            "heads": 4,
            "layers": 1,
            "dropout": 0.1,
            "bottleneck": "average2",
            "pp_layers": 3,             # Number of property predictor layers
            "ID_d_model": None,
            "ID_d_inner_hid": None,
            "ID_heads": None,
            "ID_d_k": None,
            "ID_d_v": None,
            "ID_layers": None,
            "ID_width": None,
            "num_params": None,
            "epoch_time": 0,
        }

    def __getitem__(self, param):
        if param in self._params:
            return self._params[param]
        else:
            raise Warning("Param {} unrecognised".format(param))

    def __setitem__(self, param, value):
        if not value is None:
            if param in self._params:
                self._params[param] = value
            else:
                print("Param {} unrecognised".format(param))

    def load(self, fn):
        with open(fn, mode='rb') as f:
            self._params = pickle.load(f)

    def save(self, fn):
        with open(fn, mode='wb') as f:
            pickle.dump(self._params, f, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self._params)

    def setIDparams(self):
        # Will by default set interim decoder parameters
        # to have same as normal parameters
        if "ar" in self._params["bottleneck"]:
            for key in self._params:
                if "ID" in key:
                    if self._params[key] is None:
                        self._params[key] = self._params[key.replace("ID_", "")]

    def dump(self):
        # get max length
        m_len = max([len(key) for key in self._params])
        for key in self._params:
            if "ID" in key and "ar" not in self._params["bottleneck"]:
                pass
            else:
                print("\t{}  {}".format(key.ljust(m_len), self._params[key]))

    def dumpToCSV(self, filename):
        # Check if csv already exists
        if not os.path.exists(filename):
            arr = np.transpose(np.array(list(self._params.items())))
            rownum = 1
        else:
            arr = np.genfromtxt(filename, delimiter=",", dtype=str)
            # check if this model already exists
            rownum = np.where(arr[:, 0] == self["model"])[0]
            if not rownum:
                newvals = [self._params[key] for key in self._params]
                num_pad = np.shape(arr)[1] - len(newvals)
                [newvals.extend("-") for _ in range(num_pad)]
                arr = np.vstack((arr, newvals))
                rownum = np.shape(arr)[0] - 1
            else:
                rownum = rownum[0]

        np.savetxt(filename, arr, delimiter=",", fmt='%s')

        return rownum, arr

    def equals(self, other_params):
        for key in self._params:
            if other_params[key] != self._params[key]:
                return False
        return True

    @property
    def params(self):
        return self._params


class DefaultDecoderParams(AttnParams):
    def __init__(self):
        super().__init__()
        self["layers"] = 4
        self["d_model"] = 160
        self["d_inner_hid"] = 1024
        self["d_k"] = 16
        self["d_v"] = 16
        self["heads"] = 10


class TokenList:
    def __init__(self, token_list):
        self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>']
        self.id2t.extend(token_list)
        self.t2id = {v: k for k, v in enumerate(self.id2t)}

    def id(self, x):
        return self.t2id.get(x, 1)

    def token(self, x):
        return self.id2t[x]

    def num(self):
        return len(self.id2t)

    def startid(self):
        return 2

    def endid(self):
        return 3

    def tokenize(self, input_str):
        '''
        Given a string input sequence will return a tokenised sequence
        :param input_str:
        :return:
        '''
        if isinstance(input_str, str):
            src_seq = np.zeros((1, len(input_str) + 3), dtype='int32')
            src_seq[0, 0] = self.startid()
            for i, z in enumerate(input_str):
                src_seq[0, 1 + i] = self.id(z)
            src_seq[0, len(input_str) + 1] = self.endid()
        else:
            src_seq = np.expand_dims(input_str, 0)

        return src_seq

    def onehotify(self, input_str, pad_length = None):
        '''
        Given a string input sequence will return a tokenised sequence
        :param input_str:
        :return:
        '''
        src_seq = self.tokenize(input_str)
        src_seq = to_categorical(src_seq, self.num())
        if pad_length:
            src_seq = np.pad(src_seq, ((0,0), (0, pad_length - np.shape(src_seq)[1]), (0,0)), mode='constant')
        return src_seq
