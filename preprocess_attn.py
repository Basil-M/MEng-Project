import argparse
import pandas
import h5py
import numpy as np
from molecules.utils import one_hot_array, one_hot_index

from sklearn.model_selection import train_test_split

MAX_NUM_ROWS = 1000
SMILES_COL_NAME = 'structure'
INFILE='data/zinc12.h5'
OUTFILE='data/processed.txt'
#INFILE='../keras-molecules/data/zinc12.h5'

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

    # Get chemical structures from dataset
    structures = data[args.smiles_column]

    #Write structures to output
    f = open(args.outfile, 'w')
    for struct in structures:
        f.write(struct + "\n") # + "\t" + struct + "\n")
        # f.write(struct + "\t" + struct + "\n")

if __name__ == '__main__':
    main()
