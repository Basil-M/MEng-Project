from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe

from hyperas import optim

from keras.optimizers import Adam
from molecules.transformer import LRSchedulerPerStep
from hyperas.distributions import quniform, choice
from molecules.model import TriTransformer
import dataloader as dd
from utils import WeightAnnealer_epoch

MODEL_ARCH = 'TRANSFORMER'
MODEL_NAME = 'avg_model'
MODEL_DIR = 'models/'


# d_file = 'data/zinc_100k.txt'
# tokens = dd.MakeSmilesDict(d_file, dict_file='data/SMILES_dict.txt')

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

    d_file = 'data/zinc_100k.txt'
    tokens = dd.MakeSmilesDict(d_file, dict_file='data/SMILES_dict.txt')
    data_train, data_test, props_train, props_test = dd.MakeSmilesData(d_file, tokens=tokens,
                                                                       h5_file=d_file.replace('.txt',
                                                                                              '_data.h5'))
    x_train = [data_train, props_train]
    y_train = None
    x_test = [data_test, props_test]
    y_test = None
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    params = dd.AttnParams()
    params.set("latent_dim", 296)
    params.set("bottleneck", "average")
    params.set("kl_pretrain_epochs", 1)
    params.set("kl_anneal_epochs", 2)
    params.set("ID_width", 4)

    # model params to change
    d_model = {{quniform(64, 512, 4)}}
    d_inner_hid = {{quniform(128, 2048, 4)}}
    d_k = {{quniform(4, 100, 1)}}
    layers = {{quniform(1, 7, 1)}}
    warmup = {{quniform(6000, 20000, 100)}}

    params.set("d_model", int(d_model))
    params.set("d_inner_hid", int(d_inner_hid))
    params.set("d_k", int(d_k))
    params.set("layers", int(layers))

    # Automatically set params from above
    params.set("d_v", params.get("d_k"))
    params.set("heads", int(np.ceil(d_model / d_k)))

    params.setIDparams()
    # GET TOKENS
    d_file = 'data/zinc_100k.txt'
    tokens = dd.MakeSmilesDict(d_file, dict_file='data/SMILES_dict.txt')
    model = TriTransformer(tokens, p=params)

    cb = []
    cb.append(LRSchedulerPerStep(params.get("d_model"), warmup=int(warmup)))

    cb.append(WeightAnnealer_epoch(model.kl_loss_var,
                                   anneal_epochs=params.get("kl_anneal_epochs"),
                                   max_val=params.get("kl_max_weight"),
                                   init_epochs=params.get("kl_pretrain_epochs")))

    model.compile_vae(Adam(0.001, 0.9, 0.98, epsilon=1e-9, clipnorm=1.0, clipvalue=0.5))

    result = model.autoencoder.fit(x_train, None, batch_size=25,
                                   epochs=3,
                                   validation_data=(x_test, None),
                                   callbacks=cb)

    # get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_accu'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model.autoencoder}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
