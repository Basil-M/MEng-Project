from __future__ import print_function

from os import getenv

import numpy as np
from hyperas import optim
from hyperas.distributions import quniform
from hyperopt import Trials, STATUS_OK, tpe

import utils
from model_analysis import property_distributions as PD
from train import trainTransformer

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

    d_file = 'data/zinc_100k.h5'
    data_train, data_test, props_train, props_test, tokens = utils.load_dataset(d_file, "TRANSFORMER", True)

    x_train = [data_train, data_train, props_train]
    y_train = None
    x_test = [data_test, data_test, props_test]
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
    ## NUMBER OF GPUS
    # GET NUMBER OF AVAILABLE GPUS
    CUDA_VISIBLE_DEVICES = getenv('CUDA_VISIBLE_DEVICES')
    if CUDA_VISIBLE_DEVICES is None:
        N_GPUS = 1
    else:
        N_GPUS = len(CUDA_VISIBLE_DEVICES.split(","))

    ## PARAMETERS
    params = utils.AttnParams()
    params["latent_dim"] = 60
    params["bottleneck"] = "average1"
    params["kl_pretrain_epochs"] = 1
    params["kl_anneal_epochs"] = 1
    params["batch_size"] = 50
    params["epochs"] = 4

    # model params to change
    d_model = {{quniform(60, 150, 4)}}
    d_inner_hid = {{quniform(128, 2048, 4)}}
    d_k = {{quniform(4, 30, 2)}}
    layers = {{quniform(1, 5, 1)}}

    params["d_model"] = int(d_model)
    params["d_inner_hid"] = int(d_inner_hid)
    params["d_k"] = int(d_k)
    params["layers"] = int(layers)
    params["pp_weight"] = 1.25
    # Automatically set params from above
    params["d_v"] = params["d_k"]
    params["heads"] = int(np.ceil(d_model / d_k))
    params.setIDparams()
    # GET TOKENS
    _, _, _, _, tokens = utils.load_dataset('data/zinc_100k.h5', "TRANSFORMER", True)

    model, result = trainTransformer(params, tokens=tokens, data_train=x_train, data_test=x_test,
                                     callbacks=["var_anneal"])

    # get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    output = PD(x_train[0], x_train[-1],
                num_seeds=200,
                num_decodings=3,
                model=model,
                beam_width=5)

    frac_valid = output["num_valid"] / output["num_mols"]
    print("With params:")
    params.dump()
    print("Validation acc:", validation_acc, "Fraction valid:", frac_valid)

    return {'loss': -frac_valid, 'status': STATUS_OK, 'model': model.autoencoder}


if __name__ == '__main__':
    np.random.seed(170)
    #    with supress_stderr():
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
