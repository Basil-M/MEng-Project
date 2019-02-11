import argparse
import pandas
import h5py
import numpy as np
from rdkit import Chem
from utils import one_hot_array, one_hot_index
from functools import reduce
from sklearn.model_selection import train_test_split

MAX_NUM_ROWS = 3000
SMILES_COL_NAME = 'structure'
INFILE = 'data/zinc12.h5'
OUTFILE = 'data/zinc_3k.txt'


def get_arguments():
    parser = argparse.ArgumentParser(description='Prepare data for training')

    parser.add_argument('--infile', type=str, help='Input file name',
                        default=INFILE)
    parser.add_argument('--outfile', type=str, help='Output file name',
                        default=OUTFILE)
    parser.add_argument('--length', type=int, metavar='N', default=MAX_NUM_ROWS,
                        help='Maximum number of rows to include (randomly sampled).')
    parser.add_argument('--smiles_column', type=str, default=SMILES_COL_NAME,
                        help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    parser.add_argument('--property_column', type=str,
                        help="Name of the column that contains the property values to predict. Default: None")
    parser.add_argument('--model_arch', type=str, metavar='N', default='VAE',
                        help='Model architecture to use - options are VAE, TRANSFORMER')
    parser.add_argument('--train_frac', type=float, metavar='0.8', default=0.8,
                        help='Fraction of data to use for training')

    return parser.parse_args()


def chunk_iterator(dataset, chunk_size=1000):
    chunk_indices = np.array_split(np.arange(len(dataset)),
                                   len(dataset) / chunk_size)
    for chunk_ixs in chunk_indices:
        chunk = dataset[chunk_ixs]
        yield (chunk_ixs, chunk)
    raise StopIteration


def main():
    args = get_arguments()

    data = pandas.read_hdf(args.infile, 'table')
    keys = data[args.smiles_column].map(len) < 121

    if args.length <= len(keys):
        data = data[keys].sample(n=args.length)
    else:
        data = data[keys]

    structures = data[args.smiles_column].map(lambda x: list(x.ljust(120)))

    # Currently transformer model takes data in a different format
    # TODO(Basil): Merge how Transformer/VAE handle data
    if args.model_arch == 'TRANSFORMER':
        # Write structures to output
        f = open(args.outfile, 'w')
        for struct in structures:
            # canonicalise
            struct = ''.join(struct)
            #print(struct)
            mol = Chem.MolFromSmiles(struct)
            if mol:
                struct = Chem.MolToSmiles(mol)
                f.write(struct + "\n")
    else:
        if args.property_column:
            properties = data[args.property_column][keys]

        del data

        # Get training and testing indices
        train_idx, test_idx = map(np.array,
                                  train_test_split(structures.index, test_size=1-args.train_frac))

        charset = list(reduce(lambda x, y: set(y) | x, structures, set()))

        one_hot_encoded_fn = lambda row: map(lambda x: one_hot_array(x, len(charset)),
                                             one_hot_index(row, charset))

        h5f = h5py.File(args.outfile, 'w')
        h5f.create_dataset('charset', data=charset)

        def create_chunk_dataset(h5file, dataset_name, dataset, dataset_shape,
                                 chunk_size=1000, apply_fn=None):
            new_data = h5file.create_dataset(dataset_name, dataset_shape,
                                             chunks=tuple([chunk_size] + list(dataset_shape[1:])))
            for (chunk_ixs, chunk) in chunk_iterator(dataset):
                if not apply_fn:
                    new_data[chunk_ixs, ...] = chunk
                else:
                    new_data[chunk_ixs, ...] = apply_fn(chunk)

        create_chunk_dataset(h5f, 'data_train', train_idx,
                             (len(train_idx), 120, len(charset)),
                             apply_fn=lambda ch: np.array(map(one_hot_encoded_fn,
                                                              structures[ch])))
        create_chunk_dataset(h5f, 'data_test', test_idx,
                             (len(test_idx), 120, len(charset)),
                             apply_fn=lambda ch: np.array(map(one_hot_encoded_fn,
                                                              structures[ch])))

        # calculate properties



        if args.property_column:
            h5f.create_dataset('property_train', data=properties[train_idx])
            h5f.create_dataset('property_test', data=properties[test_idx])
            h5f.close()


if __name__ == '__main__':
    main()
