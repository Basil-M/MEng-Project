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
from dataloader import AttnParams
import molecules.transformer as tr
from molecules.transformer import debugPrint, SUM_AM


# import transformer as tr


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
        if p.get("latent_dim") is None:
            self.latent_dim = p.get("d_model")
        else:
            self.latent_dim = p.get("latent_dim")
        self.pp_layers = p.get("pp_layers")
        d_emb = self.d_model

        pos_emb = tr.Embedding(self.len_limit, d_emb, trainable=False,
                               weights=[tr.GetPosEncodingMatrix(self.len_limit, d_emb)])

        i_word_emb = tr.Embedding(i_tokens.num(), d_emb, trainable=True)
        o_word_emb = i_word_emb

        self.encoder = tr.Encoder(self.d_model, p.get("d_inner_hid"), p.get("heads"), p.get("d_k"), p.get("d_v"),
                                  p.get("layers"), p.get("dropout"),
                                  latent_dim=p.get("latent_dim"), word_emb=i_word_emb, pos_emb=pos_emb)
        self.decoder = tr.Decoder(self.d_model, p.get("d_inner_hid"), p.get("heads"), p.get("d_k"), p.get("d_v"),
                                  p.get("layers"), p.get("dropout"),
                                  latent_dim=p.get("latent_dim"), word_emb=o_word_emb, pos_emb=pos_emb)
        self.target_layer = TimeDistributed(Dense(o_tokens.num(), use_bias=False))

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def compile_vae(self, optimizer='adam', active_layers=999):
        src_seq_input = Input(shape=(None,), dtype='int32')
        tgt_seq_input = src_seq_input  # Input(shape=(None,), dtype='int32')

        src_seq = src_seq_input
        tgt_seq = Lambda(lambda x: x[:, :-1])(tgt_seq_input)
        tgt_true = Lambda(lambda x: x[:, 1:])(tgt_seq_input)

        src_pos = Lambda(self.get_pos_seq)(src_seq)
        tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
        if not self.src_loc_info: src_pos = None

        enc_output, kl_loss, z_mean, z_log_var, enc_attn = self.encoder(src_seq,
                                                                        src_pos,
                                                                        active_layers=active_layers,
                                                                        return_att=True)
        dec_output, dec_attn, encdec_attn = self.decoder(tgt_seq,
                                                         tgt_pos,
                                                         src_seq,
                                                         enc_output,
                                                         active_layers=active_layers,
                                                         return_att=True)
        final_output = self.target_layer(dec_output)

        # Property prediction
        encoded_input = Input(shape=(None, self.latent_dim), dtype='float', name='latent_rep')
        h = Dense(self.latent_dim, activation='linear')(encoded_input)
        for _ in range(self.pp_layers - 1):
            h = Dense(self.latent_dim, activation='linear')(h)
        prop_output = Dense(1, activation='linear')(h)

        def get_loss(args):
            y_pred, y_true, kl_loss = args
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

        self.autoencoder = Model(src_seq_input, loss)
        self.autoencoder.add_loss([loss])

        # For property prediction
        self.property_predictor = Model(encoded_input, prop_output)

        # For outputting next symbol
        self.output_model = Model(src_seq_input, final_output)

        # For encoding to z
        self.output_latent = Model(src_seq_input, enc_output)

        # For getting attentions
        attn_list = enc_attn + dec_attn + encdec_attn
        self.output_attns = Model(src_seq_input, attn_list)
        self.autoencoder.compile(optimizer, None)
        self.autoencoder.metrics_names.append('ppl')
        self.autoencoder.metrics_tensors.append(self.ppl)
        self.autoencoder.metrics_names.append('accu')
        self.autoencoder.metrics_tensors.append(self.accu)
        self.make_fast_decode_model()

    def make_src_seq_matrix(self, input_seq):
        '''
        Given a string input sequence will return a tokenised sequence
        :param input_seq:
        :return:
        '''
        src_seq = np.zeros((1, len(input_seq) + 3), dtype='int32')
        src_seq[0, 0] = self.i_tokens.startid()
        for i, z in enumerate(input_seq):
            src_seq[0, 1 + i] = self.i_tokens.id(z)
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

        enc_output, _, _, _ = self.encoder(src_seq, src_pos)
        self.encode_model = Model(src_seq_input, enc_output)

        enc_ret_input = Input(shape=(None, self.latent_dim))
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        final_output = self.target_layer(dec_output)

        # Only need to run encoder once then run decoder several times
        # So take latent rep as an input
        self.decode_model = Model([src_seq_input, enc_ret_input, tgt_seq_input], final_output)

        self.encode_model.compile('adam', 'mse')
        self.decode_model.compile('adam', 'mse')

    def decode_sequence_fast(self, input_seq, delimiter=''):
        '''
        Greedy decodes a sequence by keeping the most probable output symbol at each stage
        :param input_seq: String e.g. 'Cc1cccc1'
        :param delimiter:
        :return: output sequence as a string
        '''
        if self.decode_model is None:
            self.make_fast_decode_model()

        src_seq = self.make_src_seq_matrix(input_seq)

        # Encode input to latent representation
        latent_rep = self.encode_model.predict_on_batch(src_seq)

        decoded_tokens = []
        target_seq = np.zeros((1, self.len_limit), dtype='int32')
        target_seq[0, 0] = self.o_tokens.startid()

        for i in range(self.len_limit - 1):
            output = self.decode_model.predict_on_batch([src_seq, latent_rep, target_seq])
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


# https://github.com/Lsdefine/attention-is-all-you-need-keras
class TriTransformer:
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
        self.bottleneck = p.get("bottleneck")
        self.stddev = p.get("stddev")
        if self.bottleneck == "none":
            p.set("latent_dim", None)
            self.latent_dim = None
        elif p.get("latent_dim") is None:
            self.latent_dim = p.get("d_model")
        else:
            self.latent_dim = p.get("latent_dim")

        self.pp_layers = p.get("pp_layers")
        d_emb = self.d_model

        pos_emb = tr.Embedding(self.len_limit, d_emb, trainable=False,
                               weights=[tr.GetPosEncodingMatrix(self.len_limit, d_emb)])
        i_word_emb = tr.Embedding(i_tokens.num(), d_emb)
        o_word_emb = i_word_emb

        self.encoder = tr.VariationalEncoder(self.d_model, p.get("d_inner_hid"), p.get("heads"), p.get("d_k"),
                                             p.get("d_v"),
                                             p.get("layers"), p.get("dropout"), word_emb=i_word_emb, pos_emb=pos_emb)

        if p.get("bottleneck") == "average":
            self.encoder_to_latent = tr.AvgLatent(p.get("d_model"), p.get("latent_dim"))
        elif p.get("bottleneck") == "interim_decoder":
            latent_pos_emb = tr.Embedding(p.get("latent_dim"), p.get("ID_d_model"), trainable=True)
            self.encoder_to_latent = tr.InterimDecoder(p.get("ID_d_model"), p.get("ID_d_inner_hid"),
                                                       p.get("ID_heads"), p.get("ID_d_k"), p.get("ID_d_v"),
                                                       p.get("ID_layers"), p.get("ID_width"), p.get("dropout"),
                                                       stddev=p.get("stddev"),
                                                       latent_dim=p.get("latent_dim"),
                                                       pos_emb=latent_pos_emb)
        elif p.get("bottleneck") == "none":
            self.encoder_to_latent = tr.Vec2Variational(p.get("d_model"), self.len_limit)

        self.latent_to_decoder = tr.LatentToEmbedded(self.d_model,
                                                     latent_dim=self.latent_dim,
                                                     stddev=p.get("stddev"))

        self.decoder = tr.DecoderFromLatent(self.d_model, p.get("d_inner_hid"), p.get("heads"), p.get("d_k"),
                                            p.get("d_v"),
                                            p.get("layers"), p.get("dropout"),
                                            word_emb=o_word_emb, pos_emb=pos_emb)
        self.target_layer = TimeDistributed(Dense(o_tokens.num(), use_bias=False))

        if self.stddev == 0 or self.stddev is None:
            self.kl_loss_var = None
        else:
            self.kl_loss_var = K.variable(0.0, dtype=np.float, name='kl_loss_weight')

        if p.get("pp_weight") == 0 or p.get("pp_weight") is None:
            self.pp_loss_var = None
            print("Not joint training property encoder")
        else:
            self.pp_loss_var = K.variable(p.get("pp_weight"), dtype=np.float, name='pp_loss_weight')
            print("Joint training property encoder with PP weight {}".format(p.get("pp_weight")))

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def compile_vae(self, optimizer='adam', active_layers=999):
        src_seq_input = Input(shape=(None,), dtype='int32', name='src_seq_input')
        tgt_seq_input = src_seq_input  # Input(shape=(None,), dtype='int32', name='tgt_seq_input')

        src_seq = src_seq_input

        # pr = Lambda(lambda x: tf.Print(x, [x], "\nSRC_SEQ: ", summarize=SUM_AM))
        src_seq = debugPrint(src_seq, "SRC_SEQ")
        # pr = Lambda(lambda x: tf.Print(x, [x], "\nTGT_SEQ: ", summarize=SUM_AM))
        tgt_seq = Lambda(lambda x: x[:, :-1])(tgt_seq_input)
        tgt_seq = debugPrint(tgt_seq, "TGT_SEQ")
        # tgt_seq = pr(tgt_seq)
        tgt_true = Lambda(lambda x: x[:, 1:])(tgt_seq_input)

        src_pos = Lambda(self.get_pos_seq)(src_seq)
        src_pos = debugPrint(src_pos, "SRC_POS")

        tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
        if not self.src_loc_info: src_pos = None

        enc_output, enc_attn = self.encoder(src_seq,
                                            src_pos,
                                            active_layers=active_layers,
                                            return_att=True)

        enc_output = debugPrint(enc_output, "ENC_OUTPUT")

        if self.bottleneck == "interim_decoder":
            z_mean, z_logvar, _ = self.encoder_to_latent(src_seq, enc_output)
        else:
            z_mean, z_logvar = self.encoder_to_latent(enc_output)

        # z_logvar = pr(z_logvar)
        print("Finished setting up decoder.")

        z_sampled, dec_input = self.latent_to_decoder(z_mean, z_logvar)

        dec_output, dec_attn, encdec_attn = self.decoder(tgt_seq,
                                                         tgt_pos,
                                                         z_sampled,
                                                         dec_input,
                                                         active_layers=active_layers,
                                                         return_att=True)

        dec_output = debugPrint(dec_output, "DEC_OUTPUT")

        final_output = self.target_layer(dec_output)

        # Property prediction
        if self.latent_dim is None:
            latent_dim = self.d_model * self.len_limit
            latent_vec = Input(shape=[self.d_model, self.len_limit], dtype='float', name='latent_rep')
        else:
            latent_dim = self.latent_dim
            latent_vec = Input(shape=[latent_dim], dtype='float', name='latent_rep')

        if self.pp_loss_var is not None:
            # latent_vec = Lambda(lambda x: K.reshape(x, [-1, latent_dim]))(z_sampled)
            latent_vec = z_sampled

        h = Dense(latent_dim, input_shape=(latent_dim,), activation='linear')(latent_vec)
        num_props = 4
        prop_input = Input(shape=[num_props])
        for _ in range(self.pp_layers - 1):
            h = Dense(latent_dim, activation='linear')(h)
        prop_output = Dense(num_props, activation='linear')(h)

        # RECONSTRUCTION LOSS
        def rec_loss(args):
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            return K.sum(loss)

        # KL DIVERGENCE LOSS
        def kl_loss(args):
            z_mean_, z_log_var_ = args

            return - 0.5 * self.kl_loss_var * tf.reduce_sum(1 + z_log_var_ - K.square(z_mean_) - K.exp(z_log_var_),
                                                            name='KL_loss_sum')

        # PROPERTY PREDICTION LOSS
        def pp_loss(args):
            prop_input_, prop_output_ = args
            SE = K.pow(prop_input_ - prop_output_, 2)
            SE = tf.reduce_sum(SE)
            return self.pp_loss_var * K.sqrt(SE)

            # def get_loss(args):
            #     y_pred, y_true, z_mean_, z_log_var_ = args
            #
            #     # RECONSTRUCTION LOSS
            #     y_true = tf.cast(y_true, 'int32')
            #     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            #     mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            #     loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            #     loss = K.mean(loss)
            #
            #     # z_mean_ = tf.Print(z_mean_, [z_mean_], "\nLOSS CALCS: z_mean_: ", summarize=SUM_AM)
            #     # z_log_var_ = tf.Print(z_log_var_, [z_log_var_], "\nLOSS CALCS: z_log_var_: ", summarize=SUM_AM)
            #
            #     # KL LOSS
            #     if self.stddev == 0 or self.stddev is None:
            #         print("No VAE loss!")
            #     else:
            #         if self.stddev == 0.01:
            #             kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_var_ - K.square(z_mean_) - K.exp(z_log_var_),
            #                                              name='KL_loss_sum')
            #         else:
            #             kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var_ - K.square(z_mean_) - K.exp(z_log_var_),
            #                                             name='KL_loss_sum')
            #         loss += self.kl_loss_var*kl_loss
            #     return loss

        def get_accu(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        losses = []
        losses.append(Lambda(rec_loss, name='ReconstructionLoss')([final_output, tgt_true]))

        kl = None
        if self.kl_loss_var is not None:
            kl = Lambda(kl_loss, name='VariationalLoss')([z_mean, z_logvar])
            losses.append(kl)

        pp = None
        if self.pp_loss_var is not None:
            pp = Lambda(pp_loss, name='PropertyLoss')([prop_input, prop_output])
            losses.append(pp)

        loss = Lambda(tf.reduce_sum)(losses)
        # loss = Lambda(get_loss, name='LossFn')([final_output, tgt_true, z_mean, z_logvar])

        self.ppl = Lambda(K.exp)(loss)
        self.accu = Lambda(get_accu)([final_output, tgt_true])

        # For encoding to z
        self.output_latent = Model(src_seq_input, z_sampled)

        if self.pp_loss_var is None:
            self.autoencoder = Model(src_seq_input, loss)
        else:
            self.autoencoder = Model([src_seq_input, prop_input], loss)

        self.autoencoder.add_loss([loss])

        # For property prediction
        # if not self.latent_dim is None:
        # self.property_predictor = Model(latent_vec, prop_output)

        # For outputting next symbol
        self.output_model = Model(src_seq_input, final_output)

        # For getting attentions
        attn_list = enc_attn + dec_attn + encdec_attn
        self.output_attns = Model(src_seq_input, attn_list)
        self.autoencoder.compile(optimizer, None)
        self.autoencoder.metrics_names.append('ppl')
        self.autoencoder.metrics_tensors.append(self.ppl)
        self.autoencoder.metrics_names.append('accu')
        self.autoencoder.metrics_tensors.append(self.accu)

        if pp is not None:
            self.autoencoder.metrics_names.append('pp_loss')
            self.autoencoder.metrics_tensors.append(pp)

        if kl is not None:
            self.autoencoder.metrics_names.append('kl_loss')
            self.autoencoder.metrics_tensors.append(kl)
        # self.autoencoder.metrics_names.append('meanmean')
        # self.autoencoder.metrics_tensors.append(self.meanz)

        # if self.stddev != 0 and self.stddev is not None:
        #     self.autoencoder.metrics_names.append('meanlogvar')
        #     self.meanzvar = Lambda(tf.reduce_mean)(z_logvar)
        #     self.autoencoder.metrics_tensors.append(self.meanzvar)

        # session = K.get_session()
        # for layer in self.autoencoder.layers:
        #     if hasattr(layer, 'kernel_initializer'):
        #         print("Running kernel initialiser for {}".format(layer.name))
        #         layer.kernel.initializer.run(session=session)
        # self.autoencoder.summary()
        # self.make_fast_decode_model()

    def make_src_seq_matrix(self, input_seq):
        '''
        Given a string input sequence will return a tokenised sequence
        :param input_seq:
        :return:
        '''
        src_seq = np.zeros((1, len(input_seq) + 3), dtype='int32')
        src_seq[0, 0] = self.i_tokens.startid()
        for i, z in enumerate(input_seq):
            src_seq[0, 1 + i] = self.i_tokens.id(z)
        src_seq[0, len(input_seq) + 1] = self.i_tokens.endid()
        return src_seq

    def decode_sequence(self, input_seq, delimiter=''):
        src_seq = self.make_src_seq_matrix(input_seq)
        decoded_tokens = []

        # First get the latent representation

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
        src_seq = Input(shape=(None,), dtype='int32')
        latent_seq_input = Input(shape=(None,), dtype='float')
        tgt_seq = Input(shape=(None,), dtype='int32')
        src_pos = Lambda(self.get_pos_seq)(src_seq)
        tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
        if not self.src_loc_info: src_pos = None

        # Decode latent representation
        enc_output, _, _, _ = self.encoder(src_seq, src_pos)
        self.encode_model = Model(src_seq, enc_output)

        enc_ret_input = Input(shape=(None, self.latent_dim))
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        final_output = self.target_layer(dec_output)

        # Only need to run encoder once then run decoder several times
        # So take latent rep as an input
        self.decode_model = Model([src_seq, enc_ret_input, tgt_seq], final_output)

        self.encode_model.compile('adam', 'mse')
        self.decode_model.compile('adam', 'mse')

    def decode_sequence_fast(self, input_seq, delimiter=''):
        '''
        Greedy decodes a sequence by keeping the most probable output symbol at each stage
        :param input_seq: String e.g. 'Cc1cccc1'
        :param delimiter:
        :return: output sequence as a string
        '''
        if self.decode_model is None:
            self.make_fast_decode_model()

        src_seq = self.make_src_seq_matrix(input_seq)

        # Encode input to latent representation
        latent_rep = self.encode_model.predict_on_batch(src_seq)

        decoded_tokens = []
        target_seq = np.zeros((1, self.len_limit), dtype='int32')
        target_seq[0, 0] = self.o_tokens.startid()

        for i in range(self.len_limit - 1):
            output = self.decode_model.predict_on_batch([src_seq, latent_rep, target_seq])
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
