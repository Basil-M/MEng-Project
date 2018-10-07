import random, os, sys
import numpy as np
from keras.models import *
from keras.backend import int_shape as sh
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from keras.objectives import binary_crossentropy
import tensorflow as tf

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
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None, d_h = None, stddev=0.01):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

        if d_h is None:
            self.d_h = d_model
        else:
            self.d_h = d_h

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


        # Sampling function
        def sampling(args):
            z_mean_, z_log_var_ = args

            batch_size = K.shape(z_mean_)[0]
            seq_length = K.shape(z_mean_)[1]
            epsilon = K.random_normal(shape=(batch_size,seq_length, self.d_h), mean=0., stddev=self.stddev)
            # z_mean_ = tf.Print(z_mean_, [tf.shape(z_mean_)])
            # epsilon = tf.Print(epsilon, [tf.shape(epsilon)])

            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon


        # Set up bottleneck

        z_mean = Dense(self.d_h, name='z_mean', activation='linear')(h)
        z_log_var = Dense(self.d_h, name='z_log_var', activation='linear')(h)
        z_samp = Lambda(sampling, name='lambda')([z_mean, z_log_var])
        # kl_loss = 0
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # kl_loss = tf.Print(kl_loss,[kl_loss])
        # kl_loss = tf.reduce_sum(kl_loss, -1)

        return (z_samp, kl_loss, z_mean, z_log_var, atts) if return_att else (z_samp, kl_loss, z_mean, z_log_var)

def _buildEncoder(x, src_pos, params,
                  word_emb=None, pos_emb=None,
                  return_att = False, active_layers = 999):
    # Get relevant parameters
    d_model = params.get("d_model")
    d_h = params.get("d_h")
    d_inner_hid = params.get("d_inner_hid")
    n_head = params.get("n_head")
    d_k = params.get("d_k")
    d_v = params.get("d_v")
    max_length = params.get("len_limit")
    layers = params.get("layers")
    dropout = params.get("dropout")
    epsilon_std = params.get("epsilon")

    # Word embedding of source sequence
    h = word_emb(x)

    # Positional embedding
    if pos_emb is not None:
        pos = pos_emb(src_pos)
        h = Add()([h, pos])

    # Set up list for returning attentions
    if return_att:
        atts = []

    # Set up encoder layers
    layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    mask = Lambda(lambda x: GetPadMask(x, x))(x)
    for enc_layer in layers[:active_layers]:
        h, att = enc_layer(h, mask)
        if return_att:
            atts.append(att)

    if d_h is None:
        # This means there is no bottle neck
        # But still set up two dense layers which will find z_mean and z_log_var
        d_h = d_model

    # Sampling function
    def sampling(args):
        z_mean_, z_log_var_ = args
        batch_size = K.shape(z_mean_)[0]
        epsilon = K.random_normal(shape=(batch_size, d_h), mean=0., stddev=epsilon_std)
        return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

    # Set up bottleneck
    z_mean = Dense(d_h, name='z_mean', activation='linear')(h)
    z_log_var = Dense(d_h, name='z_log_var', activation='linear')(h)

    # # VAE Loss function
    # def vae_loss(x, x_decoded_mean):
    #     x = K.flatten(x)
    #     x_decoded_mean = K.flatten(x_decoded_mean)
    #
    #     xent_loss = max_length * binary_crossentropy(x, x_decoded_mean)
    #     kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    #     return xent_loss + kl_loss

    # return vae_loss, Lambda(sampling, output_shape=(d_h,), name='lambda')([z_mean, z_log_var])
    z_samp = Lambda(sampling, output_shape=(d_h,), name='lambda')([z_mean, z_log_var])

    return z_samp, z_mean, z_log_var
def _buildDecoder(z, tgt_seq, tgt_pos, dec_so_far,
                  params, charset_length,
                  word_emb=None, pos_emb=None,
                  return_att=False, active_layers=999):
    # Get relevant parameters
    d_model = params.get("d_model")
    d_inner_hid = params.get("d_inner_hid")
    n_head = params.get("n_head")
    d_k = params.get("d_k")
    d_v = params.get("d_v")
    layers = params.get("layers")
    dropout = params.get("dropout")

    # Word embedding of sequence decoded thus far
    h = word_emb(tgt_seq)

    #Add to positional embedding
    if pos_emb is not None:
        pos = pos_emb(tgt_pos)
        h = Add()([h, pos])


    #Set up self padding mask so it doesn't look in the future
    self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(tgt_seq)
    self_sub_mask = Lambda(GetSubMask)(tgt_seq)
    self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
    enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([tgt_seq, dec_so_far])


    # Set up list for returning attentions
    if return_att:
        self_atts, enc_atts = [], []

    # Set up decoder layers
    layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    for dec_layer in layers[:active_layers]:
        h, self_att, enc_att = dec_layer(h, z, self_mask, enc_mask)
        if return_att:
            self_atts.append(self_att)
            enc_atts.append(enc_att)

    # Dense layer to vocabulary size for probability of next decoded symbol
    decoded_mean = TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)
    return (decoded_mean, self_atts, enc_atts) if return_att else decoded_mean

class Decoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None, d_h = None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]
        if not d_h is None:
            self.bridge = Dense(d_model, input_shape=(d_h, ))
        else:
            self.bridge = None

    def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output, return_att=False, active_layers=999):
        if not self.bridge is None:
            enc_output = self.bridge(enc_output)

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

class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        self.basic = d_model ** -0.5
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
