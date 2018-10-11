# Attention is All You Need (to propose novel chemical designs)
This work aims to adapt [Google's solely attention-based architecture](https://arxiv.org/abs/1706.03762) to a variational auto-encoder, with the end goal of novel chemical generation. Please note this is a **work in progress**.

## Previous work
As mentioned, it is primarily based on Google's transformer architecture. It builds upon [preceding work](https://arxiv.org/abs/1610.02415) by Aspuru-Guzik et al; by using a string representation of molecules, they were able to leverage an NLP approach to convert the discrete representation of molecules to a continuous latent representation. A property predictor is trained on this latent representation. A continuous latent representation allows for efficient search and optimisation techniques, opening avenues for numerous methods for generating novel chemical structures.

## Credits
This work builds on [Max Hodak's implementation](https://github.com/maxhodak/keras-molecules) and [Lsdefine's implementation](https://github.com/Lsdefine/attention-is-all-you-need-keras) of the Transformer architecture.

## Requirements

Install using `pip install -r requirements.txt` or build a docker container: `docker build .`

The docker container can also be built different TensorFlow binary, for example in order to use GPU:

`docker build --build-arg TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl .`

You'll need to ensure the proper CUDA libraries are installed for this version to work.

## Getting the datasets

A small 50k molecule dataset is included in `data/smiles_50k.h5` to make it easier to get started playing around with the model. A much larger 500k ChEMBL 21 extract is also included in `data/smiles_500k.h5`. A model trained on `smiles_500k.h5` is included in `data/model_500k.h5`.

All h5 files in this repo by [git-lfs](https://git-lfs.github.com/) rather than included directly in the repo.

To download original datasets to work with, you can use the `download_dataset.py` script:

* `python download_dataset.py --dataset zinc12`
* `python download_dataset.py --dataset chembl22`
* `python download_dataset.py --uri http://my-domain.com/my-file.csv --outfile data/my-file.csv`

## Preparing the data

To train the network you need a lot of SMILES strings. The `preprocess.py` script assumes you have an HDF5 file that contains a table structure, one column of which is named `structure` and contains one SMILES string no longer than 120 characters per row. The script then:

- Normalizes the length of each string to 120 by appending whitespace as needed.
- Builds a list of the unique characters used in the dataset. (The "charset")
- Substitutes each character in each SMILES string with the integer ID of its location in the charset.
- Converts each character position to a one-hot vector of len(charset).
- Saves this matrix to the specified output file.

Example:

`python preprocess.py data/smiles_50k.h5 data/processed.h5`

## Training the network

The preprocessed data can be fed into the `train.py` script:

`python train.py data/processed.h5 model.h5 --epochs 20`

If a model file already exists it will be opened and resumed. If it doesn't exist, it will be created.

By default, the latent space is 292-D per the paper, and is configurable with the `--latent_dim` flag. If you use a non-default latent dimensionality don't forget to use `--latent_dim` on the other scripts (eg `sample.py`) when you operate on that model checkpoint file or it will be confused.

## Sampling from a trained model

The `sample.py` script can be used to either run the full autoencoder (for testing) or either the encoder or decoder halves using the `--target` parameter. The data file must include a charset field.

Examples:

```
python sample.py data/processed.h5 model.h5 --target autoencoder

python sample.py data/processed.h5 model.h5 --target encoder --save_h5 encoded.h5

python sample.py target/encoded.h5 model.h5 --target decoder
```

## Performance

