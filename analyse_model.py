from tensor2tensor.visualization import attention
import h5py
import numpy as np
from sample_latent import visualize_latent_rep
from molecules.model import Transformer
from dataloader import SmilesToArray, AttnParams, MakeSmilesDict, MakeSmilesData
from keras.optimizers import Adam
from metakernel.display import display
from utils import LoadList
from rdkit import Chem
from IPython import embed

IBUPROFEN_SMILES = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
ZINC_RANDOM = 'C[NH+](CC1CCCC1)[C@H]2CCC[C@@H](C2)O'


def call_html():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))


def visualise_latents(latents_file):
    with h5py.File(latents_file) as dfile:
        z_test = dfile['z_test'][:]
        # z_test, z_train = dfile['z_test'][:], dfile['z_train'][:]
    z_test = np.mean(z_test, axis=1)
    visualize_latent_rep(z_test, tsne_perplexity=60, pca_components=50, tsne_components=3)


def visualise_attention(smiles_str, attn_model: Transformer):
    '''
    Script for visualising the attention in the model for a given input
    :param smiles_str: SMILES string input e.g. 'Cc1ccccc1'
    :param attn_model: a loaded Transformer model
    :param params: params of that transformer model
    :param tokens: tokens of that transformer model (charset)
    :return:
    '''
    tokens = attn_model.i_tokens
    smiles_seq = SmilesToArray(smiles_str, tokens=tokens, max_len=attn_model.len_limit)
    attentions = attn_model.output_attns.predict_on_batch([smiles_seq, smiles_seq])

    # split attentions into individual attentions
    n_layers = int(len(attentions) / 3)
    reshape_attn = lambda ATT: np.swapaxes(np.expand_dims(np.array(ATT), 1), 3, 4)
    enc_atts = reshape_attn(attentions[0:n_layers])
    dec_atts = reshape_attn(attentions[n_layers:2 * n_layers])
    encdec_atts = reshape_attn(attentions[2 * n_layers:])

    print("Shapes of arrays:\n\tenc_atts:\t{}\n\tdec_atts:\t{}\n\tencdec_atts:\t{}".format(np.shape(enc_atts),
                                                                                           np.shape(dec_atts),
                                                                                           np.shape(encdec_atts)))
    call_html()

    out_str = list(smiles_str) + [tokens.token(tokens.endid())]
    in_str = [tokens.token(tokens.startid())] + out_str

    attention.show(in_str, out_str, enc_atts, dec_atts, encdec_atts)


def property_distributions(test_data, num_seeds, num_decodings, attn_model: Transformer, beam_width=1):
    # Get random seeds from test_data
    indices = np.array(range(0, len(test_data)))
    np.random.shuffle(indices)
    indices = indices[0:num_seeds]
    test_data = np.array(test_data)  # allow for fancy indexing
    test_data = list(test_data[indices])

    output_molecules = []
    # decode molecules multiple times
    for dec_itr in range(num_decodings):
        print(dec_itr)
        # decode molecules
        if beam_width == 1:
            output_itr = [attn_model.decode_sequence_fast(seq) for seq in test_data]
        else:
            output_itr = []
            for seq in test_data:
                output_itr += [s[0] for s in attn_model.beam_search(seq, beam_width)]

        # only keep if it's unique
        for output in output_itr:
            if output not in output_molecules:
                output_molecules.append(output)

    # get qed vals of input data
    qed_vals = [Chem.QED.qed(Chem.MolFromSmiles(molecule)) for molecule in test_data]

    gen_qed_vals = []
    for molecule in output_molecules:
        mol = Chem.MolFromSmiles(molecule)
        # mol is None if it wasn't a valid SMILES string
        if mol is not None:
            gen_qed_vals.append(Chem.QED.qed(mol))

    print("Generated {} unique sequences, of which {} were valid.".format(len(output_molecules), len(gen_qed_vals)))

    return np.mean(qed_vals), np.std(qed_vals), np.mean(gen_qed_vals), np.std(gen_qed_vals), len(gen_qed_vals) / len(
        test_data)


def main():
    model_dir = 'models/VA_64/'

    # Load in model
    model_params = AttnParams()
    model_params.load(model_dir + "params.pkl")

    # Get data
    d_file = model_params.get("d_file")
    data = [d for d in LoadList(d_file)]  # List of SMILES strings
    tokens = MakeSmilesDict(d_file, dict_file=d_file.replace('.txt', '_dict.txt'))

    # Prepare model
    model = Transformer(tokens, model_params)
    model.compile_vae(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
    model.autoencoder.load_weights(model_dir + "best_model.h5")

    # Test random molecule
    print(IBUPROFEN_SMILES)
    # print(model.beam_search(IBUPROFEN_SMILES,5)[0][0])
    model.encode_model.load_weights(model_dir + "best_model.h5", by_name=True)
    model.decode_model.load_weights(model_dir + "best_model.h5", by_name=True)
    s = model.beam_search(IBUPROFEN_SMILES,5)
    [print(seq[0]) for seq in s]
    # print(s)
    # print(model.beam_search(IBUPROFEN_SMILES,5)[1][0])

    data_mu, data_sigma, gen_mu, gen_sigma, frac_valid = property_distributions(data,
                                                                                num_seeds=100,
                                                                                num_decodings=5,
                                                                                attn_model=model,
                                                                                beam_width=2)

    print("Data QED:\t {:.2f} ± {:.2f}".format(data_mu, data_sigma))
    print("Gen QED:\t {:.2f} ± {:.2f}".format(gen_mu, gen_sigma))
    print("Valid mols:\t {:.2f}".format(frac_valid))

    # visualise_attention('Cc1ccccc1', model, model_params, tokens)
    # visualise_latents(model_dir + 'latents.h5')


if __name__ == '__main__':
    main()
