import copy
import tensorflow as tf
import numpy as np

from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
from dataloader import attn_params
import transformer as tr


class MoleculeVAE():
    autoencoder = None

    def create(self,
               charset,
               max_length=120,
               latent_rep_size=292,
               weights_file=None):
        charset_length = len(charset)

        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, z)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        x1 = Input(shape=(max_length, charset_length))
        vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        self.autoencoder = Model(
            x1,
            self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)

        self.autoencoder.compile(optimizer='Adam',
                                 loss=vae_loss,
                                 metrics=['accuracy'])

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std=0.01):
        h = Convolution1D(9, 9, activation='relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation='relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences=True, name='gru_1')(h)
        h = GRU(501, return_sequences=True, name='gru_2')(h)
        h = GRU(501, return_sequences=True, name='gru_3')(h)
        return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)

    def load(self, charset, weights_file, latent_rep_size=292):
        self.create(charset, weights_file=weights_file, latent_rep_size=latent_rep_size)


# https://github.com/Lsdefine/attention-is-all-you-need-keras
class Transformer:
    def __init__(self, i_tokens, p, o_tokens=None):
        self.i_tokens = i_tokens

        if o_tokens is None:
            # Autoencoder
            o_tokens = i_tokens

        self.o_tokens = o_tokens
        self.len_limit = p.get("len_limit")
        self.src_loc_info = True
        self.d_model = p.get("d_model")
        self.decode_model = None
        self.latent_dim = p.get("latent_dim")
        self.pp_layers = p.get("pp_layers")
        d_emb = self.d_model

        pos_emb = tr.Embedding(self.len_limit, d_emb, trainable=False,
                               weights=[tr.GetPosEncodingMatrix(self.len_limit, d_emb)])

        i_word_emb = tr.Embedding(i_tokens.num(), d_emb)
        o_word_emb = i_word_emb

        self.encoder = tr.Encoder(self.d_model, p.get("d_inner_hid"), p.get("n_head"), p.get("d_k"), p.get("d_v"),
                                  p.get("layers"), p.get("dropout"),
                                  latent_dim=p.get("latent_dim"), word_emb=i_word_emb, pos_emb=pos_emb)
        self.decoder = tr.Decoder(self.d_model, p.get("d_inner_hid"), p.get("n_head"), p.get("d_k"), p.get("d_v"),
                                  p.get("layers"), p.get("dropout"),
                                  latent_dim=p.get("latent_dim"), word_emb=o_word_emb, pos_emb=pos_emb)
        self.target_layer = TimeDistributed(Dense(o_tokens.num(), use_bias=False))

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def compile_vae(self, optimizer='adam', active_layers=999):
        src_seq_input = Input(shape=(None,), dtype='int32')
        tgt_seq_input = Input(shape=(None,), dtype='int32')

        src_seq = src_seq_input
        tgt_seq = Lambda(lambda x: x[:, :-1])(tgt_seq_input)
        tgt_true = Lambda(lambda x: x[:, 1:])(tgt_seq_input)

        src_pos = Lambda(self.get_pos_seq)(src_seq)
        tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
        if not self.src_loc_info: src_pos = None

        enc_output, kl_loss, z_mean, z_log_var = self.encoder(src_seq, src_pos, active_layers=active_layers)
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, active_layers=active_layers)
        final_output = self.target_layer(dec_output)

        # Property prediction
        encoded_input = Input(shape=(None, self.latent_dim), dtype='float', name='latent_rep')
        h = Dense(self.latent_dim, activation='linear')(encoded_input)
        for _ in range(self.pp_layers - 1):
            h = Dense(self.latent_dim, activation='linear')(h)
        prop_output = Dense(1, activation='linear')(h)

        def get_loss(args):
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            loss = K.mean(loss)

            return loss

        def get_accu(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        loss = Lambda(get_loss)([final_output, tgt_true])
        self.ppl = Lambda(K.exp)(loss)
        self.accu = Lambda(get_accu)([final_output, tgt_true])

        self.autoencoder = Model([src_seq_input, tgt_seq_input], loss)
        self.autoencoder.add_loss([loss])

        # For property prediction
        self.property_predictor = Model(encoded_input, prop_output)

        # For outputting next symbol
        self.output_model = Model([src_seq_input, tgt_seq_input], final_output)

        # For encoding to z
        self.output_latent = Model([src_seq_input, tgt_seq_input], enc_output)

        self.autoencoder.compile(optimizer, None)
        self.autoencoder.metrics_names.append('ppl')
        self.autoencoder.metrics_tensors.append(self.ppl)
        self.autoencoder.metrics_names.append('accu')
        self.autoencoder.metrics_tensors.append(self.accu)

    def make_src_seq_matrix(self, input_seq):
        src_seq = np.zeros((1, len(input_seq) + 3), dtype='int32')
        src_seq[0, 0] = self.i_tokens.startid()
        for i, z in enumerate(input_seq): src_seq[0, 1 + i] = self.i_tokens.id(z)
        src_seq[0, len(input_seq) + 1] = self.i_tokens.endid()
        return src_seq

    def decode_sequence(self, input_seq, delimiter=''):
        src_seq = self.make_src_seq_matrix(input_seq)
        decoded_tokens = []
        target_seq = np.zeros((1, self.len_limit), dtype='int32')
        target_seq[0, 0] = self.o_tokens.startid()
        for i in range(self.len_limit - 1):
            output = self.output_model.predict_on_batch([src_seq, target_seq])
            sampled_index = np.argmax(output[0, i, :])
            sampled_token = self.o_tokens.token(sampled_index)
            decoded_tokens.append(sampled_token)
            if sampled_index == self.o_tokens.endid(): break
            target_seq[0, i + 1] = sampled_index
        return delimiter.join(decoded_tokens[:-1])

    def make_fast_decode_model(self):
        src_seq_input = Input(shape=(None,), dtype='int32')
        tgt_seq_input = Input(shape=(None,), dtype='int32')
        src_seq = src_seq_input
        tgt_seq = tgt_seq_input

        src_pos = Lambda(self.get_pos_seq)(src_seq)
        tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
        if not self.src_loc_info: src_pos = None
        enc_output = self.encoder(src_seq, src_pos)
        self.encode_model = Model(src_seq_input, enc_output)

        enc_ret_input = Input(shape=(None, self.d_model))
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_ret_input)
        final_output = self.target_layer(dec_output)
        self.decode_model = Model([src_seq_input, enc_ret_input, tgt_seq_input], final_output)

        self.encode_model.compile('adam', 'mse')
        self.decode_model.compile('adam', 'mse')

    def decode_sequence_fast(self, input_seq, delimiter=''):
        if self.decode_model is None: self.make_fast_decode_model()
        src_seq = self.make_src_seq_matrix(input_seq)
        enc_ret = self.encode_model.predict_on_batch(src_seq)

        decoded_tokens = []
        target_seq = np.zeros((1, self.len_limit), dtype='int32')
        target_seq[0, 0] = self.o_tokens.startid()
        for i in range(self.len_limit - 1):
            output = self.decode_model.predict_on_batch([src_seq, enc_ret, target_seq])
            sampled_index = np.argmax(output[0, i, :])
            sampled_token = self.o_tokens.token(sampled_index)
            decoded_tokens.append(sampled_token)
            if sampled_index == self.o_tokens.endid(): break
            target_seq[0, i + 1] = sampled_index
        return delimiter.join(decoded_tokens[:-1])

    def beam_search(self, input_seq, topk=5, delimiter=''):
        if self.decode_model is None: self.make_fast_decode_model()
        src_seq = self.make_src_seq_matrix(input_seq)
        src_seq = src_seq.repeat(topk, 0)
        enc_ret = self.encode_model.predict_on_batch(src_seq)

        final_results = []
        decoded_tokens = [[] for _ in range(topk)]
        decoded_logps = [0] * topk
        lastk = 1
        target_seq = np.zeros((topk, self.len_limit), dtype='int32')
        target_seq[:, 0] = self.o_tokens.startid()
        for i in range(self.len_limit - 1):
            if lastk == 0 or len(final_results) > topk * 3: break
            output = self.decode_model.predict_on_batch([src_seq, enc_ret, target_seq])
            output = np.exp(output[:, i, :])
            output = np.log(output / np.sum(output, -1, keepdims=True) + 1e-8)
            cands = []
            for k, wprobs in zip(range(lastk), output):
                if target_seq[k, i] == self.o_tokens.endid(): continue
                wsorted = sorted(list(enumerate(wprobs)), key=lambda x: x[-1], reverse=True)
                for wid, wp in wsorted[:topk]:
                    cands.append((k, wid, decoded_logps[k] + wp))
            cands.sort(key=lambda x: x[-1], reverse=True)
            cands = cands[:topk]
            backup_seq = target_seq.copy()
            for kk, zz in enumerate(cands):
                k, wid, wprob = zz
                target_seq[kk,] = backup_seq[k]
                target_seq[kk, i + 1] = wid
                decoded_logps[kk] = wprob
                decoded_tokens.append(decoded_tokens[k] + [self.o_tokens.token(wid)])
                if wid == self.o_tokens.endid(): final_results.append((decoded_tokens[k], wprob))
            decoded_tokens = decoded_tokens[topk:]
            lastk = len(cands)
        final_results = [(x, y / (len(x) + 1)) for x, y in final_results]
        final_results.sort(key=lambda x: x[-1], reverse=True)
        final_results = [(delimiter.join(x), y) for x, y in final_results]
        return final_results
