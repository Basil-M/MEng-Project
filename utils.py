# coding = utf-8
import h5py
import numpy as np
from keras.callbacks import Callback
from keras import backend as K


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

    def __init__(self, params, param_filename):
        self._params = params
        self._filename = param_filename

    def on_epoch_end(self, epoch, logs={}):
        self._params.set("current_epoch", self._params.get("current_epoch") + 1)
        self._params.save(self._filename)
        return

    def on_train_end(self, logs={}):
        if not self._params.get("ae_trained"):
            self._params.set("current_epoch", 1)
            self._params.set("ae_trained", True)
        self._params.save(self._filename)
        return

    def epoch(self):
        return self._params.get("current_epoch")


class WeightAnnealer_epoch(Callback):
    '''Weight of variational autoencoder scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            weight for the VAE (float).
        Currently just adjust kl weight, will keep xent weight constant
    '''

    def __init__(self, var, anneal_epochs=10, max_val=1.0, init_epochs=1):
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


def load_dataset(filename, split=True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)


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
from collections import defaultdict

import os.path as op

_fscores = None


def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
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
