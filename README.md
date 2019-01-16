# Attention is All You Need (to propose novel chemical designs)
This work aims to adapt [Google's solely attention-based architecture](https://arxiv.org/abs/1706.03762) to a variational auto-encoder, with the end goal of novel chemical generation. Please note this is a **work in progress**.

## Previous work
As mentioned, it is primarily based on Google's transformer architecture. It builds upon [preceding work](https://arxiv.org/abs/1610.02415) by Aspuru-Guzik et al; by using a string representation of molecules, they were able to leverage an NLP approach to convert the discrete representation of molecules to a continuous latent representation. A property predictor is trained on this latent representation. A continuous latent representation allows for efficient search and optimisation techniques, opening avenues for numerous methods for generating novel chemical structures.

## Credits
This work builds on [Max Hodak's implementation](https://github.com/maxhodak/keras-molecules) of the Molecule Autoencoder and [Lsdefine's implementation](https://github.com/Lsdefine/attention-is-all-you-need-keras) of the Transformer architecture.

## Requirements

Install using `pip install -r requirements.txt`.

You'll need to ensure the proper CUDA libraries are installed for this version to work. It is suggested that Tensorflow and RDKit be installed via anaconda:
`conda install -c rdkit rdkit`
`conda install -c anaconda tensorflow-gpu `

## Getting the datasets

A small 50k molecule dataset is included in `data/smiles_50k.h5` to make it easier to get started playing around with the model. A much larger 500k ChEMBL 21 extract is also included in `data/smiles_500k.h5`. A model trained on `smiles_500k.h5` is included in `data/model_500k.h5`.

All h5 files in this repo by [git-lfs](https://git-lfs.github.com/) rather than included directly in the repo.

To download original datasets to work with, you can use the `download_dataset.py` script:

* `python download_dataset.py --dataset zinc12`
* `python download_dataset.py --dataset chembl22`
* `python download_dataset.py --uri http://my-domain.com/my-file.csv --outfile data/my-file.csv`

## Preparing the data

Training the network requires a subset of SMILES strings. At least 100k is suggested. This must be preprocessed before training.
The `preprocess.py` script assumes you have an HDF5 file that contains a table structure, one column of which is named `structure` and contains one SMILES string no longer than 120 characters per row.
The recursive VAE and the Transformer models take inputs in different forms, so the choice of model must be specified.

For the recursive VAE, the preprocess script will:
  
- Normalizes the length of each string to 120 by appending whitespace as needed.
- Builds a list of the unique characters used in the dataset. (The "charset")
- Substitutes each character in each SMILES string with the integer ID of its location in the charset.
- Converts each character position to a one-hot vector of len(charset).
- Saves this matrix to the specified output file.

For the Transformer, the preprocess script will:

- Canonicalise SMILES
- Save as a list of text files 

`python preprocess.py data/smiles_50k.h5 data/trans_processed.h5 --model_arch TRANSFORMER`

`python preprocess.py data/smiles_50k.h5 data/recurs_processed.h5 --model_arch VAE`

## Training the network

The training file has a large number of options corresponding to model parameters. The easiest way to figure out how to run it is to run:

`python train.py -h`

### Parameters
General parameters related to training/model definition are shown below

| Parameter      | Description                                            |
|----------------|--------------------------------------------------------|
| `--epochs`     | Number of training epochs                              |
| `--model`      | Name of the models e.g. TAvg_196                       |
| `--models_dir` | Folder containing model directories                    |
| `--data`       | Path to preprocessed data file e.g. data/zinc_250k.txt |
| `--batch_size` | Batch size                                             |
| `--latent_dim` | Dimensionality of latent space                         |
| `--stddev`     | Standard deviation of sampling in the latent space. Setting `--stddev 0` will disable variational component.    |
| `--model_arch` | Model architecture. Options are VAE and TRANSFORMER    |
| `--dropout`    | Dropout used in the models                             |

### Transformer Parameters

Setting `--model_arch TRANSFORMER` will use the transformer model, which has the following parameters:

| Parameter      | Description                                                                  |
|----------------|------------------------------------------------------------------------------|
| `--bottleneck` | Sets architecture of bottleneck. Options are `average` and `interim_decoder` |
| `--d_model`    | Dimensionality of word embeddings (and thus the entire model)                |
| `--d_k`        | Dimensionality of keys (and queries) in attention mechanisms                 |
| `--d_v`        | Dimensionality of values generated in attention mechanisms                   |
| `--heads`      | Number of attention heads to use                                             |
| `--layers`     | Number of layers                                                             |

These must be set alongside parameters above. There are currently two choices of bottleneck:

#### Averaging Bottleneck
`--bottleneck average`

Performs a weighted average over the length of the sequence to yield a vector of size `d_model` 
which is used to generate means and variances of size `d_latent`.

No extra parameters need to be specified.

#### Interim Decoder Bottleneck

The interim decoder uses a second decoder stack to recursively produce means and variances for the latent space.

This interim decoder can have different parameters to the main Encoder-Decoder stacks - 
therefore all the above parameters (except for `--bottleneck`) are repeated, prefixed with `ID`; 
e.g. `--ID_layers ` will set the number of layers in the interim decoder, and `--ID_d_k` will set its key dimensionality.

#### Property predictor 
The property predictor is a simple dense network with two parameters:

| Parameter     | Description                                                                |
|---------------|----------------------------------------------------------------------------|
| `--pp_epochs` | Number of epochs to train property predictor (only if not jointly trained) |
| `--pp_layers` | Number of property predictor layers                                        |

## Upcoming
- Joint training of property predictor with autoencoder
- Annealing of variational loss
- Model analysis scripts
- Attention mechanism visualisation scripts
- More thorough installation guide
- Sample model results & graphs
