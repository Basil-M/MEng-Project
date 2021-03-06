from __future__ import print_function

import argparse
# import os, sys
# import h5py
# import numpy as np
# #
# from molecules.model import MoleculeVAE
# from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
#     decode_smiles_from_indexes, load_dataset

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure, axes, scatter, title, show
#
# from rdkit import Chem
# from rdkit.Chem import Draw
#
# from keras.models import Sequential, Model, load_model

LATENT_DIM = 292
PCA_COMPONENTS = 50
TSNE_LEARNING_RATE = 750.0
TSNE_ITERATIONS = 1000
TSNE_COMPONENTS = 2
TSNE_PERPLEXITY = 30.0


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='HDF5 file to read input data from.')
    parser.add_argument('model', type=str, help='Trained Keras model to use.')
    parser.add_argument('--save_h5', type=str, help='Name of a file to write HDF5 output to.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--tsne_lr', metavar='LR', type=float, default=TSNE_LEARNING_RATE,
                        help='Learning to use for t-SNE.')
    parser.add_argument('--tsne_components', metavar='N', type=int, default=TSNE_COMPONENTS,
                        help='Number of components to use for t-SNE.')
    parser.add_argument('--tsne_perplexity', metavar='P', type=float, default=TSNE_PERPLEXITY)
    parser.add_argument('--tsne_iterations', metavar='N', type=int, default=TSNE_ITERATIONS)
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help='Fit manifold and render a visualization. If this flag is not used, the sampled data' +
                             ' will simply be returned with no further processing.')
    parser.add_argument('--skip-pca', dest='use_pca', action='store_false',
                        help='Skip PCA preprocessing of data to feed into t-SNE.')
    parser.add_argument('--pca_components', metavar='N', type=int, default=PCA_COMPONENTS,
                        help='Number of components to use for PCA.')
    parser.set_defaults(use_pca=True)
    parser.set_defaults(visualize=False)

    return parser.parse_args()


def visualize_latent_rep(x_latent, use_pca=True, pca_components=PCA_COMPONENTS, tsne_components=TSNE_COMPONENTS,
                         tsne_perplexity=TSNE_PERPLEXITY, tsne_lr=TSNE_LEARNING_RATE, tsne_iterations=TSNE_ITERATIONS):
    print("pca_on=%r pca_comp=%d tsne_comp=%d tsne_perplexity=%f tsne_lr=%f" % (
        use_pca,
        pca_components,
        tsne_components,
        tsne_perplexity,
        tsne_lr
    ))

    if use_pca:
        pca = PCA(n_components=pca_components)
        x_latent = pca.fit_transform(x_latent)

    # figure(figsize=(6, 6))
    # scatter(x_latent[:, 0], x_latent[:, 1], marker='.')
    # show()

    print("Running TSNE")
    tsne = TSNE(n_components=tsne_components,
                perplexity=tsne_perplexity,
                learning_rate=tsne_lr,
                n_iter=tsne_iterations,
                verbose=4)

    print("Fitting TSNE transform")
    x_latent_proj = tsne.fit_transform(x_latent)
    del x_latent

    fig = figure(figsize=(6, 6))
    print("Plotting scatter")
    if tsne_components==2:
        scatter(x_latent_proj[:, 0], x_latent_proj[:, 1], marker='.')
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_latent_proj[:, 0], x_latent_proj[:, 1], x_latent_proj[:, 2], marker='.')
    show()
#
#
# def main():
#     args = get_arguments()
#     model = MoleculeVAE()
#
#     data, data_test, charset = load_dataset(args.data)
#
#     if os.path.isfile(args.model):
#         model.load(charset, args.model, latent_rep_size=args.latent_dim)
#     else:
#         raise ValueError("Model file %s doesn't exist" % args.model)
#
#     x_latent = model.encoder.predict(data)
#     if not args.visualize:
#         if not args.save_h5:
#             np.savetxt(sys.stdout, x_latent, delimiter='\t')
#         else:
#             h5f = h5py.File(args.save_h5, 'w')
#             h5f.create_dataset('charset', data=charset)
#             h5f.create_dataset('latent_vectors', data=x_latent)
#             h5f.close()
#     else:
#         visualize_latent_rep(x_latent, args.use_pca, args.pca_components, args.tsne_components, args.tsne_perplexity,
#                              args.tsne_lr, args.tsne_iterations)
#

if __name__ == '__main__':
    main()
