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

    def __init__(self, weight, start_val = 0.1, b = 1.5, max_val = 1.0, init_epochs = 1):
        self.weight_var = weight
        self.w0 = start_val
        self.max_weight = max_val
        self.inc = b
        self.init_epochs = init_epochs
        print("Initialising weight loss annealer with w0 = {} and increment of {} each epoch".format(start_val, b))
        print("\tExpected to reach max_val of {} in {} epochs".format(max_val, int(self.init_epochs + np.ceil(np.log(max_val/start_val)/np.log(1.5)))))
        print("\tFirst training {} epochs without variational term".format(self.init_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        weight = min(self.max_weight, self.w0*(self.inc ** (epoch - self.init_epochs)))
        if epoch < self.init_epochs:
            weight = 0

        print(epoch)
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
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()


def load_dataset(filename, split = True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)