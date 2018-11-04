import random, os, sys
import numpy as np
from keras.models import *
from keras.backend import int_shape as sh
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from keras.objectives import binary_crossentropy
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences


# try:
#     from dataloader import TokenList, pad_to_longest
# # for transformer
# except:
#     pass


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big matrixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.EncoderLayappend(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)

        elif self.mode == 1:
            heads = [];
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head);
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn


class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


class DecoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None):
        output, slf_attn = self.self_att_layer(dec_input, dec_input, dec_input, mask=self_mask)
        output, enc_attn = self.enc_att_layer(output, enc_output, enc_output, mask=enc_mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn, enc_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def GetPadMask(q, k):
    '''
    Given query q and key k, returns a mask which prevents attention to padding
    :param q: Tensor query
    :param k: Tensor key
    :return: Tensor mask
    '''
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def GetSubMask(s):
    '''

    :param s:
    :return:
    '''
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    # mask = K.print_tensor(mask, "SUBMASK:")
    return mask


class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None, latent_dim=None, stddev=0.01):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

        if latent_dim is None:
            self.latent_dim = d_model
        else:
            self.latent_dim = latent_dim

        self.stddev = stddev

    def __call__(self, src_seq, src_pos, return_att=False, active_layers=999):
        h = self.emb_layer(src_seq)

        if src_pos is not None:
            pos = self.pos_layer(src_pos)
            h = Add()([h, pos])
        if return_att: atts = []
        mask = Lambda(lambda x: GetPadMask(x, x))(src_seq)
        for enc_layer in self.layers[:active_layers]:
            h, att = enc_layer(h, mask)
            if return_att: atts.append(att)

        # Set up bottleneck

        if self.stddev is None:
            # Model isn't probabilistic
            z_mean = Dense(self.latent_dim, name='z_mean', activation='linear')(h)
            z_log_var = None
            kl_loss = 0
            z_samp = z_mean
        else:
            # Include variational component
            # Sampling function
            def sampling(args):
                z_mean_, z_log_var_ = args
                batch_size = K.shape(z_mean_)[0]
                seq_length = K.shape(z_mean_)[1]
                epsilon = K.random_normal(shape=(batch_size, seq_length, self.latent_dim), mean=0., stddev=self.stddev)
                # z_mean_ = tf.Print(z_mean_, [tf.shape(z_mean_)])
                # epsilon = tf.Print(epsilon, [tf.shape(epsilon)])

                return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

            z_mean = Dense(self.latent_dim, name='z_mean', activation='linear')(h)
            z_log_var = Dense(self.latent_dim, name='z_log_var', activation='linear')(h)
            z_samp = Lambda(sampling, name='lambda')([z_mean, z_log_var])
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        return (z_samp, kl_loss, z_mean, z_log_var, atts) if return_att else (z_samp, kl_loss, z_mean, z_log_var)

class InterimEncoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None, latent_dim=None, stddev=0.01):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

        if latent_dim is None:
            self.latent_dim = d_model
        else:
            self.latent_dim = latent_dim

        self.stddev = stddev

    def __call__(self, src_seq, src_pos, return_att=False, active_layers=999):
        h = self.emb_layer(src_seq)

        if src_pos is not None:
            pos = self.pos_layer(src_pos)
            h = Add()([h, pos])
        if return_att: atts = []
        mask = Lambda(lambda x: GetPadMask(x, x))(src_seq)
        for enc_layer in self.layers[:active_layers]:
            h, att = enc_layer(h, mask)
            # h = Lambda(lambda z: K.print_tensor(z, "\n\nENCODING H: "))(h)
            if return_att: atts.append(att)

        return (h, atts) if return_att else h


class RNNDecoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None, latent_dim=None, stddev=1):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

        if latent_dim is None: latent_dim = d_model  # no bottleneck

        self.latent_embedder = Dense(d_model, activation='linear', name='latent_embedder')

        self.latent_dim = latent_dim
        self.mean_layer = Dense(1, input_shape=(d_model*latent_dim,), name='mean_layer')
        self.logvar_layer = Dense(1, input_shape=(d_model*latent_dim,), name='logvar_layer')
        self.conc_layer = Concatenate(axis=1, name='concat')

        # Sample from the means/variances decoded so far
        def sampling(args):
            z_mean_, z_logvar_ = args
            batch_size = K.shape(z_mean_)[0]
            k = K.shape(z_mean_)[1]
            epsilon = K.random_normal(shape=(batch_size, k), mean=0., stddev=stddev)
            return K.reshape(z_mean_ + K.exp(z_logvar_ / 2) * epsilon, [-1, k])

        def expand_dims(args):
            return K.expand_dims(args, axis=2)

        def collapse_dims(arg):
            return K.reshape(arg, [K.shape(arg)[0], 1])
            # return K.squeeze(arg, axis=-1)

        def pad_with_zeros(arg):
            paddings = [[0,0], [0, K.constant(self.latent_dim, dtype='int32') - K.shape(arg)[1]]]
            return tf.pad(arg, paddings, 'CONSTANT', constant_values=0.0)
            # return K.zeros(
            #     [K.shape(arg)[0], self.latent_dim - K.shape(arg)[1]],
            #     dtype="float")

        self.sampler = Lambda(sampling, name='InterimDecoderSampler')
        self.expand = Lambda(expand_dims, name='DimExpander')
        self.squeeze = Lambda(collapse_dims, name='DimCollapser')
        self.pad_zeros = Lambda(pad_with_zeros,name='ZerosForPadding')
        self.d_model = d_model
        self.resh = Lambda(lambda x: K.reshape(x, [-1, self.d_model*self.latent_dim]))

    def first_iter(self, src_seq, enc_output, return_att=False, active_layers=999):
        print("Setting up first decoder iteration")

        def gen_zeros(sample_vec):
            batch_size = K.shape(sample_vec)[0]
            return tf.keras.backend.zeros([batch_size, self.latent_dim], dtype='float', name='zeros')

        z_zero = Lambda(gen_zeros, name='z_zero')(enc_output)
        decoder_input = self.latent_embedder(self.expand(z_zero))
        z = decoder_input  # Add()([decoder_input, pos])

        # z_zero = Reshape([-1, 1])(z_zero)
        # Mask the output
        self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(z_zero)
        self_sub_mask = Lambda(GetSubMask)(z_zero)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
        enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([z_zero, src_seq])
        # self_mask = None
        # enc_mask = None

        if return_att: self_atts, enc_atts = [], []

        for dec_layer in self.layers[:active_layers]:
            z, self_att, enc_att = dec_layer(z, enc_output, self_mask, enc_mask)
            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)

        # z = self.shapeprinter(z)

        # mean1 = self.mean_layer(z)
        # mean1 = self.shapeprinter(mean1)
        # mean2 = self.mean_layer(mean1)
        # mean2 = self.shapeprinter(mean2)
        # output_mean = self.squeeze(mean2)
        # output_mean = self.shapeprinter(output_mean)

        # output_mean = self.squeeze(self.mean_layer2(self.mean_layer(z)))
        # output_logvar = self.squeeze(self.logvar_layer2(self.logvar_layer(z)))
        z = Lambda(lambda x: K.reshape(x, [-1, self.d_model*self.latent_dim]))(z)
        output_mean = self.mean_layer(z)
        output_mean = self.squeeze(output_mean)
        output_logvar = self.logvar_layer(z)
        output_logvar = self.squeeze(output_logvar)

        return (output_mean, output_logvar, self_atts, enc_atts) if return_att else (output_mean, output_logvar)

    def __call__(self, src_seq, enc_output, mean_so_far=None, logvar_so_far=None, active_layers=999):

        mean_so_far, logvar_so_far = self.first_iter(src_seq, enc_output)
        for _ in range(self.latent_dim - 1):
            print("Setting up decoder iteration")
            sampled_z = self.sampler([mean_so_far, logvar_so_far])

            # Should be vector of size [batch_size, latent_dim]
            sampled_z = self.pad_zeros(sampled_z)

            # Expand to matrix of size [batch_size, latent_dim, model_dim]
            z = self.latent_embedder(self.expand(sampled_z))

            # Mask the output
            self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(sampled_z)
            self_sub_mask = Lambda(GetSubMask)(sampled_z)
            self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
            enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([sampled_z, src_seq])


            for dec_layer in self.layers[:active_layers]:
                z, _, _ = dec_layer(z, enc_output, self_mask, enc_mask)

            z = self.resh(z)
            mean_k = self.squeeze(self.mean_layer(z))
            logvar_k = self.squeeze(self.logvar_layer(z))
            mean_so_far = self.conc_layer([mean_so_far, mean_k])
            logvar_so_far = self.conc_layer([logvar_so_far, logvar_k])



        return (mean_so_far, logvar_so_far)

class InterimDecoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None, latent_dim=None, stddev=1):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

        if latent_dim is None: latent_dim = d_model  # no bottleneck

        self.latent_embedder = Dense(d_model, activation='linear', name='latent_embedder')

        self.latent_dim = latent_dim
        self.stddev = stddev
        self.mean_layer = Dense(1, input_shape=(d_model*latent_dim,), name='mean_layer')
        self.logvar_layer = Dense(1, input_shape=(d_model*latent_dim,), name='logvar_layer')
        # self.mean_layer2 = Dense(1, input_shape=(latent_dim,), name='mean2_layer')
        # self.logvar_layer2 = Dense(1,input_shape=(latent_dim,), name='logvar2_layer')
        self.conc_layer = Concatenate(axis=1, name='concat')

        # Sample from the means/variances decoded so far
        def sampling(args):
            z_mean_, z_logvar_ = args
            batch_size = K.shape(z_mean_)[0]
            k = K.shape(z_mean_)[1]
            epsilon = K.random_normal(shape=(batch_size, k), mean=0., stddev=self.stddev)
            return K.reshape(z_mean_ + K.exp(z_logvar_ / 2) * epsilon, [-1, k])

        def expand_dims(args):
            return K.expand_dims(args, axis=2)

        def collapse_dims(arg):
            return K.reshape(arg, [K.shape(arg)[0], 1])
            # return K.squeeze(arg, axis=-1)

        def pad_with_zeros(arg):
            paddings = [[0,0], [0, K.constant(self.latent_dim, dtype='int32') - K.shape(arg)[1]]]
            return tf.pad(arg, paddings, 'CONSTANT', constant_values=0.0)
            # return K.zeros(
            #     [K.shape(arg)[0], self.latent_dim - K.shape(arg)[1]],
            #     dtype="float")

        self.sampler = Lambda(sampling, name='InterimDecoderSampler')
        self.expand = Lambda(expand_dims, name='DimExpander')
        self.squeeze = Lambda(collapse_dims, name='DimCollapser')
        self.pad_zeros = Lambda(pad_with_zeros,name='ZerosForPadding')
        self.d_model = d_model

    def first_iter(self, src_seq, enc_output, return_att=False, active_layers=999):
        print("Setting up first decoder iteration")

        def gen_zeros(sample_vec):
            batch_size = K.shape(sample_vec)[0]
            return tf.keras.backend.zeros([batch_size, self.latent_dim], dtype='float', name='zeros')

        z_zero = Lambda(gen_zeros, name='z_zero')(enc_output)
        decoder_input = self.latent_embedder(self.expand(z_zero))
        z = decoder_input  # Add()([decoder_input, pos])

        # z_zero = Reshape([-1, 1])(z_zero)
        # Mask the output
        self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(z_zero)
        self_sub_mask = Lambda(GetSubMask)(z_zero)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
        enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([z_zero, src_seq])
        # self_mask = None
        # enc_mask = None

        if return_att: self_atts, enc_atts = [], []

        for dec_layer in self.layers[:active_layers]:
            z, self_att, enc_att = dec_layer(z, enc_output, self_mask, enc_mask)
            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)

        # z = self.shapeprinter(z)

        # mean1 = self.mean_layer(z)
        # mean1 = self.shapeprinter(mean1)
        # mean2 = self.mean_layer(mean1)
        # mean2 = self.shapeprinter(mean2)
        # output_mean = self.squeeze(mean2)
        # output_mean = self.shapeprinter(output_mean)

        # output_mean = self.squeeze(self.mean_layer2(self.mean_layer(z)))
        # output_logvar = self.squeeze(self.logvar_layer2(self.logvar_layer(z)))
        z = Lambda(lambda x: K.reshape(x, [-1, self.d_model*self.latent_dim]))(z)
        output_mean = self.mean_layer(z)
        output_mean = self.squeeze(output_mean)
        output_logvar = self.logvar_layer(z)
        output_logvar = self.squeeze(output_logvar)

        return (output_mean, output_logvar, self_atts, enc_atts) if return_att else (output_mean, output_logvar)

    def __call__(self, src_seq, enc_output, mean_so_far=None, logvar_so_far=None, return_att=False, active_layers=999):
        print("Setting up decoder iteration")

        sampled_z = self.sampler([mean_so_far, logvar_so_far])

        # Should be vector of size [batch_size, latent_dim]
        # sampled_z = self.conc_layer([sampled_z, self.pad_zeros(sampled_z)])
        sampled_z = self.pad_zeros(sampled_z)

        # Expand to matrix of size [batch_size, latent_dim, model_dim]
        decoder_input = self.latent_embedder(self.expand(sampled_z))

        # Add positional encoding
        # pos = self.pos_layer(decoder_input)
        z = decoder_input #Add()([decoder_input, pos])

        # Mask the output
        self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(sampled_z)
        self_sub_mask = Lambda(GetSubMask)(sampled_z)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
        enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([sampled_z, src_seq])

        if return_att: self_atts, enc_atts = [], []

        for dec_layer in self.layers[:active_layers]:
            z, self_att, enc_att = dec_layer(z, enc_output, self_mask, enc_mask)
            # z = Lambda(lambda z: K.print_tensor(z, "\tINTERIM Z: "))(z)
            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)

        # mean_k = self.squeeze(self.mean_layer2(self.mean_layer(z)))
        # logvar_k = self.squeeze(self.logvar_layer2(self.logvar_layer(z)))

        z = Lambda(lambda x: K.reshape(x, [-1, self.d_model*self.latent_dim]))(z)
        mean_k = self.squeeze(self.mean_layer(z))
        logvar_k = self.squeeze(self.logvar_layer(z))
        output_mean = self.conc_layer([mean_so_far, mean_k])
        output_logvar = self.conc_layer([logvar_so_far, logvar_k])

        return (output_mean, output_logvar, self_atts, enc_atts) if return_att else (output_mean, output_logvar)

class LatentToEmbedded():
    def __init__(self, d_model, latent_dim, stddev=1):
        self.expander_layer = Dense(d_model, input_shape=(1,))
        self.stddev = stddev
        self.latent_dim = latent_dim
        self.cheating = Dense(latent_dim)

        def sampling(args):
            z_mean_, z_logvar_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, self.latent_dim), mean=0., stddev=self.stddev)
            return K.reshape(z_mean_ + K.exp(z_logvar_ / 2) * epsilon, [batch_size, self.latent_dim, 1])

        self.sampler = Lambda(sampling, name='LatentToEmbeddingSampler')

    def __call__(self, z_mean, z_logvar):
        # Sample from the means/variances decoded so far
        sampled_z = self.sampler([z_mean, z_logvar])
        K.print_tensor(sampled_z, "SAMPLED_Z = ")

        expanded_z = self.expander_layer(self.cheating(sampled_z))
        K.print_tensor(expanded_z, "EXPANDED_Z = ")

        return expanded_z #self.expander_layer(sampled_z)

class DecoderFromInterim():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output, return_att=False, active_layers=999):

        dec = self.emb_layer(tgt_seq)
        pos = self.pos_layer(tgt_pos)
        x = Add()([dec, pos])

        self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(tgt_seq)
        self_sub_mask = Lambda(GetSubMask)(tgt_seq)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])

        enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([tgt_seq, src_seq])

        if return_att: self_atts, enc_atts = [], []

        for dec_layer in self.layers[:active_layers]:
            x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)

        return (x, self_atts, enc_atts) if return_att else x


class Decoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None, latent_dim=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]
        if not latent_dim is None:
            self.bridge = Dense(d_model, input_shape=(latent_dim,))
        else:
            self.bridge = None

    def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output, return_att=False, active_layers=999):
        if self.bridge is not None:
            enc_output = self.bridge(enc_output)

        dec = self.emb_layer(tgt_seq)
        pos = self.pos_layer(tgt_pos)
        x = Add()([dec, pos])

        self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(tgt_seq)
        self_sub_mask = Lambda(GetSubMask)(tgt_seq)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
        #TODO(Basil) Use encoder mask that actually matches dimensions from interim decoder
        enc_mask = None #Lambda(lambda x: GetPadMask(x[0], x[1]))([tgt_seq, src_seq])

        if return_att: self_atts, enc_atts = [], []

        for dec_layer in self.layers[:active_layers]:
            x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)

        return (x, self_atts, enc_atts) if return_att else x


class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        self.basic = (d_model ** -0.5)
        self.warm = warmup ** -1.5
        self.step_num = 0

    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


class LRSchedulerPerEpoch(Callback):
    def __init__(self, d_model, warmup=4000, num_per_epoch=1000):
        self.basic = d_model ** -0.5
        self.warm = warmup ** -1.5
        self.num_per_epoch = num_per_epoch
        self.step_num = 1

    def on_epoch_begin(self, epoch, logs=None):
        self.step_num += self.num_per_epoch
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


class AddPosEncoding:
    def __call__(self, x):
        _, max_len, d_emb = K.int_shape(x)
        pos = GetPosEncodingMatrix(max_len, d_emb)
        x = Lambda(lambda x: x + pos)(x)
        return x


add_layer = Lambda(lambda x: x[0] + x[1], output_shape=lambda x: x[0])


# use this because keras may get wrong shapes with Add()([])
class QANet_ConvBlock:
    def __init__(self, dim, n_conv=2, kernel_size=7, dropout=0.1):
        self.convs = [SeparableConv1D(dim, kernel_size, activation='relu', padding='same') for _ in range(n_conv)]
        self.norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        for i in range(len(self.convs)):
            z = self.norm(x)
            if i % 2 == 0: z = self.dropout(z)
            z = self.convs[i](z)
            x = add_layer([x, z])
        return x


class QANet_Block:
    def __init__(self, dim, n_head, n_conv, kernel_size, dropout=0.1, add_pos=True):
        self.conv = QANet_ConvBlock(dim, n_conv=n_conv, kernel_size=kernel_size, dropout=dropout)
        self.self_att = MultiHeadAttention(n_head=n_head, d_model=dim,
                                           d_k=dim // n_head, d_v=dim // n_head,
                                           dropout=dropout, use_norm=False)
        self.feed_forward = PositionwiseFeedForward(dim, dim, dropout=dropout)
        self.norm = LayerNormalization()
        self.add_pos = add_pos

    def __call__(self, x, mask):
        if self.add_pos: x = AddPosEncoding()(x)
        x = self.conv(x)
        z = self.norm(x)
        z, _ = self.self_att(z, z, z, mask)
        x = add_layer([x, z])
        z = self.norm(x)
        z = self.feed_forward(z)
        x = add_layer([x, z])
        return x


class QANet_Encoder:
    def __init__(self, dim=128, n_head=8, n_conv=2, n_block=1, kernel_size=7, dropout=0.1, add_pos=True):
        self.dim = dim
        self.n_block = n_block
        self.conv_first = SeparableConv1D(dim, 1, padding='same')
        self.enc_block = QANet_Block(dim, n_head=n_head, n_conv=n_conv, kernel_size=kernel_size,
                                     dropout=dropout, add_pos=add_pos)

    def __call__(self, x, mask):
        if K.int_shape(x)[-1] != self.dim:
            x = self.conv_first(x)
        for i in range(self.n_block):
            x = self.enc_block(x, mask)
        return x

#
# if __name__ == '__main__':
#     itokens = TokenList(list('0123456789'))
#     otokens = TokenList(list('0123456789abcdefx'))
#
#
#     def GenSample():
#         x = random.randint(0, 99999)
#         y = hex(x);
#         x = str(x)
#         return x, y
#
#
#     X, Y = [], []
#     for _ in range(100000):
#         x, y = GenSample()
#         X.append(list(x))
#         Y.append(list(y))
#
#     X, Y = pad_to_longest(X, itokens), pad_to_longest(Y, otokens)
#     print(X.shape, Y.shape)
#
#     s2s = Transformer(itokens, otokens, 10, 15)
#     lr_scheduler = LRSchedulerPerStep(256, 4000)
#     s2s.compile('adam')
#     s2s.model.summary()
#
#
#     class TestCallback(Callback):
#         def on_epoch_end(self, epoch, logs=None):
#             print('\n')
#             for test in [123, 13245, 33467]:
#                 ret = s2s.decode_sequence(str(test))
#                 print(test, ret, hex(test))
#             print('\n')
#
#
#     TestCallback().on_epoch_end(1)
#
#     # s2s.model.load_weights('model.h5')
#     s2s.model.fit([X, Y], None, batch_size=256, epochs=40,
#                   validation_split=0.05,
#                   callbacks=[TestCallback(), lr_scheduler])
#     s2s.model.save_weights('model.h5')
