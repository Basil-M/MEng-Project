import tensorflow as tf
from keras.callbacks import *
from keras.initializers import *
from keras.layers import *

SUM_AM = 200
DEBUG = False


def debugPrint(tensor, msg):
    if DEBUG and SUM_AM != 0:
        f = Lambda(lambda x: tf.Print(tensor, [tensor], "\n{}: ".format(msg), summarize=SUM_AM))
        return f(tensor)
    else:
        return tensor


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
            heads = []
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
    Returns a mask which hides future tokens thereby preserving decoder causality
    :param s: Sequence of size [batch size, length]
    :return:
    '''
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    # creates lower triangular matrix
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, src_seq, src_pos, return_att=False, active_layers=999):
        h = self.emb_layer(src_seq)

        if src_pos is not None:
            pos = self.pos_layer(src_pos)
            h = Add()([h, pos])
        if return_att: atts = []
        mask = Lambda(lambda x: GetPadMask(x, x))(src_seq)

        # mask = pr2(mask)
        for enc_layer in self.layers[:active_layers]:
            h, att = enc_layer(h, mask)
            # h = Lambda(lambda z: K.print_tensor(z, "\n\nENCODING H: "))(h)
            if return_att: atts.append(att)

        return (h, atts) if return_att else h


NUM_LAYERS = 5
ACT = 'relu'


class AvgLatent():
    def __init__(self, d_model, latent_dim):
        self.layers = [TimeDistributed(Dense(d_model, input_shape=(d_model,), activation='relu')) for _ in range(1)]
        self.avg = Lambda(lambda x: tf.reduce_sum(x, axis=1))
        self.attn = [Dense(d_model, input_shape=(d_model,), activation='relu'),
                     Dense(1, input_shape=(d_model,), activation='linear')]
        self.after_avg = Dense(latent_dim, input_shape=(latent_dim,))
        self.mean_layer = Dense(latent_dim, input_shape=(latent_dim,), name='mean_layer')
        self.logvar_layer = Dense(latent_dim, input_shape=(latent_dim,), name='logvar_layer')

    def __call__(self, encoder_output):
        # encoder output should be [batch size, length, d_model]
        h = encoder_output

        for layer in self.layers:
            h = layer(h)

        a_vals = self.attn[1](self.attn[0](h))

        a_vals = Softmax(axis=1)(a_vals)
        h = Dot(axes=1)([a_vals, h])
        h = Lambda(lambda x: K.squeeze(x, 1))(h)
        h = self.after_avg(h)
        return self.mean_layer(h), self.logvar_layer(h)


class SumLatent():
    def __init__(self, d_model, latent_dim):
        self.layers = [TimeDistributed(Dense(d_model, input_shape=(d_model,), activation='relu')) for _ in range(1)]
        self.avg = Lambda(lambda x: tf.reduce_sum(x, axis=1))
        self.attn = [Dense(d_model, input_shape=(d_model,), activation='relu'),
                     Dense(1, input_shape=(d_model,), activation='linear')]
        self.after_avg = Dense(latent_dim, input_shape=(latent_dim,))
        self.mean_layer = Dense(latent_dim, input_shape=(latent_dim,), name='mean_layer')

        # Summing can give very large variances
        # Which impedes training
        # Restrict variances with a tanh
        self.logvar_layer1 = Dense(latent_dim, input_shape=(latent_dim,), activation='tanh', name='logvar_layer_tanh')
        self.logvar_layer = Dense(latent_dim, input_shape=(latent_dim,), name='logvar_layer')

    def __call__(self, encoder_output):
        # encoder output should be [batch size, length, d_model]
        h = encoder_output
        for layer in self.layers:
            h = layer(h)

        a_vals = self.attn[1](self.attn[0](h))

        h = Dot(axes=1)([a_vals, h])

        h = Lambda(lambda x: K.squeeze(x, 1))(h)
        h = self.after_avg(h)

        return self.mean_layer(h), self.logvar_layer(self.logvar_layer1(h))


class SumLatent2():
    def __init__(self, d_model, latent_dim):
        self.attn_mechanism = KQV_Attn(d_in=d_model, d_out=latent_dim, activation='linear', use_softmax=False)
        self.after_avg = Dense(latent_dim, input_shape=(latent_dim,), activation='relu')
        self.mean_layer = Dense(latent_dim, input_shape=(latent_dim,), name='mean_layer')
        self.logvar_layer1 = Dense(latent_dim, input_shape=(latent_dim,), activation='tanh', name='logvar_layer_tanh')
        self.logvar_layer = Dense(latent_dim, input_shape=(latent_dim,), name='logvar_layer')

    def __call__(self, encoder_output):
        # encoder output should be [batch size, length, d_model]
        h = self.attn_mechanism(encoder_output)
        h = self.after_avg(h)
        h_l = self.logvar_layer1(h)
        return self.mean_layer(h), self.logvar_layer(h_l)


class KQV_Attn():
    def __init__(self, d_in, d_out, activation='linear', use_softmax=True, scale_key=False):
        self.keys = TimeDistributed(Dense(d_out, input_shape=(d_in,), activation=activation))
        self.queries = TimeDistributed(Dense(d_out, input_shape=(d_out,), activation=activation))
        self.values = TimeDistributed(Dense(d_out, input_shape=(d_out,), activation=activation))
        self.softmax = use_softmax
        self.scale_key = scale_key
        self.d = d_in

    def __call__(self, x):
        key = self.keys(x)
        if self.scale_key:
            key /= np.sqrt(self.d)

        query = self.queries(x)
        value = self.values(x)
        attn = Lambda(lambda x: tf.matrix_diag_part(K.batch_dot(x[0], x[1], axes=2)))([key, query])
        if self.softmax:
            attn = Softmax()(attn)

        outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)))([attn, value])

        return outp


class AvgLatent2():
    def __init__(self, d_model, latent_dim):
        self.attn_mechanism = KQV_Attn(d_in=d_model, d_out=latent_dim, activation='linear', use_softmax=True)
        self.after_avg = Dense(latent_dim, input_shape=(latent_dim,), activation='relu')
        self.mean_layer = Dense(latent_dim, input_shape=(latent_dim,), name='mean_layer')
        self.logvar_layer = Dense(latent_dim, input_shape=(latent_dim,), name='logvar_layer')

    def __call__(self, encoder_output):
        # encoder output should be [batch size, length, d_model]
        h = self.attn_mechanism(encoder_output)
        h = self.after_avg(h)
        return self.mean_layer(h), self.logvar_layer(h)


class AvgLatent4():
    def __init__(self, d_model, latent_dim, heads=4):
        d_kqv = int(np.ceil(d_model / heads))
        self.attns = [KQV_Attn(d_model, d_kqv, activation='linear') for _ in range(heads)]
        self.after_avg = Dense(latent_dim, input_shape=(d_kqv * heads,), activation='relu')
        self.mean_layer = Dense(latent_dim, input_shape=(latent_dim,), name='mean_layer')
        self.logvar_layer = Dense(latent_dim, input_shape=(latent_dim,), name='logvar_layer')

    def __call__(self, encoder_output):
        # encoder output should be [batch size, length, d_model]
        h = [layer(encoder_output) for layer in self.attns]
        h = Concatenate(axis=-1)(h)
        h = self.after_avg(h)
        return self.mean_layer(h), self.logvar_layer(h)


class AvgLatent3():
    def __init__(self, d_model, latent_dim):
        self.key = Dense(latent_dim, input_shape=(d_model,), activation='linear')
        self.queries = TimeDistributed(Dense(latent_dim, input_shape=(d_model,), activation='linear'))
        self.values = TimeDistributed(Dense(latent_dim, input_shape=(d_model,), activation='linear'))
        self.after_avg = Dense(latent_dim, input_shape=(latent_dim,), activation='relu')
        self.mean_layer = Dense(latent_dim, input_shape=(latent_dim,), name='mean_layer')
        self.logvar_layer = Dense(latent_dim, input_shape=(latent_dim,), name='logvar_layer')

    def __call__(self, encoder_output):
        # encoder output should be [batch size, length, d_model]
        h = encoder_output
        k_val = Lambda(lambda x: K.expand_dims(x[:, 0, :], axis=1))(h)
        key = self.key(k_val)
        query = self.queries(h)
        value = self.values(h)
        a_vals = Dot(axes=2)([key, query])  # [batch_size, length, 1]
        a_vals = Lambda(lambda x: K.squeeze(x, axis=1))(a_vals)
        a_vals = Softmax(axis=1)(a_vals)
        h = Dot(axes=1)([a_vals, value])  # [batch_size, 1, latent_dim]
        h = self.after_avg(h)
        return self.mean_layer(h), self.logvar_layer(h)


class ConvEncoder():
    def __init__(self, layers, min_filt_size, min_filt_num, latent_dim, dense_dim, word_emb):
        self.word_emb = word_emb
        self.layers = []
        for i in range(layers):
            d = min_filt_size + i
            n = min_filt_num + i
            self.layers.append(Convolution1D(d, n, activation='relu'))

        self.after = Dense(dense_dim, activation='relu')
        self.mean_layer = Dense(latent_dim, name='z_mean')
        self.logvar_layer = Dense(latent_dim, name='z_logvar')

    def __call__(self, x):
        h = self.word_emb(x)
        for layer in self.layers:
            h = layer(h)

        h = Flatten(name='flatten_1')(h)
        h = self.after(h)
        return self.mean_layer(h), self.logvar_layer(h), None


class GRUEncoder():
    def __init__(self, layers, d_gru, latent_dim, attn=False, word_emb=None):
        self.word_emb = word_emb
        self.layers = [GRU(d_gru, return_sequences=True) for _ in range(layers)]
        if attn:
            self.combine = KQV_Attn(d_gru, latent_dim, use_softmax=True)
        else:
            self.combine = Lambda(lambda x: x[:, -1, :])
        self.after = Dense(latent_dim, activation='relu')
        self.mean_layer = Dense(latent_dim, name='z_mean')
        self.logvar_layer = Dense(latent_dim, name='z_logvar')

    def __call__(self, h):
        if self.word_emb:
            h = self.word_emb(h)

        for layer in self.layers:
            h = layer(h)

        h = self.combine(h)
        h = self.after(h)
        return self.mean_layer(h), self.logvar_layer(h), None


class Vec2Variational():
    '''
    Simply yields a mean and variance using a linear transformation
    '''

    def __init__(self, d_model, max_len):
        self.mean_layers = [TimeDistributed(Dense(d_model, input_shape=(d_model,), activation='linear')) for _ in
                            range(NUM_LAYERS)]
        self.logvar_layers = [TimeDistributed(Dense(d_model, input_shape=(d_model,), activation='linear')) for _ in
                              range(NUM_LAYERS)]

    def __call__(self, h):
        # src_seq not used; just included to match
        # calling structure of other decoders
        mean = h
        logvar = h
        for layer in self.mean_layers:
            mean = layer(mean)

        for layer in self.logvar_layers:
            logvar = layer(logvar)
        return mean, logvar


method = "iter"


class InterimDecoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, decoder_width=1, dropout=0.1, word_emb=None, pos_emb=None, latent_dim=None, stddev=1,
                 false_emb=None):
        self.latent_dim = latent_dim
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

        self.width = decoder_width

        # False embedder to go from latent dim to what decoder expects
        if false_emb is None:
            # Don't share embedder with AK
            self.false_embedder = FalseEmbeddingsTD(d_model, 1)
        else:
            self.false_embedder = false_emb

        # Calculates single attention values for each vector
        self.attn_dec = TimeDistributed(Dense(1, input_shape=(d_model,), activation='linear'))

        # Preprocesses weighted-average decoder output
        self.mean_layers = [Dense(d_model, input_shape=(d_model,), activation=ACT) for _ in range(NUM_LAYERS - 1)]
        self.logvar_layers = [Dense(d_model, input_shape=(d_model,), activation=ACT) for _ in range(NUM_LAYERS - 1)]

        # Calculate 'decoder_width' means/variances from output
        self.mean_layer = Dense(decoder_width, input_shape=(d_model,), activation=ACT, name='mean_layer')
        self.logvar_layer = Dense(decoder_width, input_shape=(d_model,), activation=ACT, name='logvar_layer')

        if method != "iter":
            self.attn = TimeDistributed(Dense(1, input_shape=(d_model,), activation='linear'))
            self.first_mean = Dense(decoder_width, input_shape=(d_model,))
            self.first_var = Dense(decoder_width, input_shape=(d_model,))

        def sampling(args):
            z_mean_, z_logvar_ = args
            if stddev is None or stddev == 0:
                # Multiply logvar by 0 so that gradient is not None
                # TODO(Basil): Fix this hack
                zero_logv = Multiply()([z_logvar_, K.zeros_like(z_logvar_)])
                return Add()([z_mean_, zero_logv])
            else:
                batch_size = K.shape(z_mean_)[0]
                k = K.shape(z_mean_)[1]
                epsilon = K.random_normal(shape=(batch_size, k), mean=0., stddev=stddev)
                return K.reshape(z_mean_ + K.exp(z_logvar_ / 2) * epsilon, [-1, k])

        self.sampler = Lambda(sampling, name='InterimDecoderSampler')
        self.expand = Lambda(lambda x: K.expand_dims(x, axis=2), name='DimExpander')
        self.squeeze = Lambda(lambda x: K.reshape(x, [K.shape(x)[0], decoder_width]), name='DimCollapser')

        if method == "iter":
            self.ldim = K.constant(latent_dim + decoder_width, dtype='int32')
        else:
            self.ldim = K.constant(latent_dim, dtype='int32')

    def __call__(self, src_seq, enc_output):
        '''
        Will iteratively compute means and variances
        :param src_seq:
        :param enc_output:
        :return:
        '''
        mean_init, var_init, z_init = self.first_iter(src_seq, enc_output)

        def the_loop(args):
            z_mean_, z_logvar_, z_ = args

            # use interim decoder to generate latent dimension iteratively
            z_mean, z_logvar, z, _, _ = tf.while_loop(self.cond, self.step,
                                                      [z_mean_, z_logvar_, z_, src_seq, enc_output],
                                                      shape_invariants=[tf.TensorShape([None, None]),
                                                                        tf.TensorShape([None, None]),
                                                                        tf.TensorShape([None, None]),
                                                                        src_seq.get_shape(),
                                                                        enc_output.get_shape()])

            # will generate too many latent dimensions, so clip it
            if method == "iter":
                z_mean = z_mean[:, -self.latent_dim:]
                z_logvar = z_logvar[:, -self.latent_dim:]
                z = z[:, -self.latent_dim:]
            else:
                z_mean = z_mean[:, :self.latent_dim]
                z_logvar = z_logvar[:, :self.latent_dim]
                z = z[:, :self.latent_dim]

            return [z_mean, z_logvar, K.reshape(z, [-1, self.latent_dim])]

        return Lambda(the_loop)([mean_init, var_init, z_init])

    def compute_next_z(self, src_seq, enc_output, z_so_far):
        '''
        Given latent values generated so far, will generate next decoder_width
        latent means and variances
        :param src_seq:     Embedded source sequence
        :param enc_output:  Output of encoder
        :param z_so_far:    Sampled latent variables generated so far
        :return:
        '''
        # z is [batch_size, k]
        # Embed to be of size [batch_size, k, d_model]
        z = self.false_embedder(z_so_far)

        # Positional embedding
        z_pos = Lambda(self.get_pos_seq)(z_so_far)
        z_pos = self.pos_layer(z_pos)
        z = Add()([z, z_pos])

        # Mask the output
        self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(z_so_far)
        self_sub_mask = Lambda(GetSubMask)(z_so_far)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
        enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([z_so_far, src_seq])

        # if return_att: self_atts, enc_atts = [], []

        # Run through interim decoder
        for dec_layer in self.layers:
            z, self_att, enc_att = dec_layer(z, enc_output, self_mask, enc_mask)
            # if return_att:
            #     self_atts.append(self_att)
            #     enc_atts.append(enc_att)

        # Decoder output is also [batch_size, k, d_model]
        # Use weighted average mechanism to produce [batch_size, d_model]
        a_vals = self.attn_dec(z)
        a_vals = Softmax(axis=1)(a_vals)
        z = Dot(axes=1)([a_vals, z])
        z = Lambda(lambda x: K.squeeze(x, axis=1))(z)

        z_mean = z
        for layer in self.mean_layers:
            z_mean = layer(z_mean)
        # Output 'width' means
        output_mean = self.mean_layer(z_mean)
        output_mean = self.squeeze(output_mean)

        z_logvar = z
        for layer in self.logvar_layers:
            z_logvar = layer(z_logvar)

        # Output 'width' logvars
        output_logvar = self.logvar_layer(z_logvar)
        output_logvar = self.squeeze(output_logvar)

        # Sample the new means/variances
        output_z = self.sampler([output_mean, output_logvar])

        return output_mean, output_logvar, output_z

    def first_iter(self, src_seq, enc_output, return_att=False, active_layers=999):
        print("Setting up first decoder iteration")

        def gen_zeros(sample_vec):
            batch_size = K.shape(sample_vec)[0]
            z0 = tf.keras.backend.random_normal([batch_size, self.width], mean=0.0, stddev=0.01,
                                                dtype='float')

            # Obfuscate shape to counter bug in TimeDistributed
            # This z0 is the first time the FalseEmbedder will be used
            # It has known shape [?, width]; so the FalseEmbedder expects this shape, but it should
            # be able to take [?, ?] as an input...
            z0 = K.reshape(z0, shape=[batch_size, -1])
            return z0
            # return tf.keras.backend.zeros([batch_size, self.decoder_width], dtype='float', name='zeros')

        # initialise
        z_zero = Lambda(gen_zeros, name='z_zero')(enc_output)

        output_mean, output_logvar, output_z = self.compute_next_z(src_seq, enc_output, z_zero)

        # This method calculates the first 'width' elements of the latent space
        # using the weighted average approach
        if method != "iter":
            a_vals = self.attn(enc_output)
            a_vals = Softmax(axis=1)(a_vals)
            h = Dot(axes=1)([a_vals, enc_output])
            h = Lambda(lambda x: K.squeeze(x, axis=1))(h)
            # h = self.squeeze(h)     # h is [batch_size, d_model]

            # Multiply output mean/var by 0
            # compute_next_z must be run at least once
            # else the variables in the layers won't be initialised
            output_mean = Multiply()([output_mean, Lambda(K.zeros_like)(output_mean)])
            output_logvar = Multiply()([output_mean, Lambda(K.zeros_like)(output_logvar)])
            output_mean = Add()([output_mean, self.first_mean(h)])
            output_logvar = Add()([output_logvar, self.first_var(h)])

            # Sample the new means/variances
            output_z = self.sampler([output_mean, output_logvar])

        return (output_mean, output_logvar, output_z)
        # return (output_mean, output_logvar, sampled_output, self_atts, enc_atts) if return_att else (
        # output_mean, output_logvar, sampled_output)

    def cond(self, mean_so_far, logvar_so_far, z_so_far, src_seq, enc_output):
        # Return true while mean length is less than latent dim
        return tf.less(K.shape(mean_so_far)[1], self.ldim)

    def step(self, mean_so_far, logvar_so_far, z_so_far, src_seq, enc_output):
        mean_so_far = debugPrint(mean_so_far, "DECODER ITERATION, MEAN SO FAR:")

        # z_so_far = self.sampler([mean_so_far, logvar_so_far])
        z_so_far = debugPrint(z_so_far, "\tSampled value:")

        mean_k, logvar_k, z_k = self.compute_next_z(src_seq, enc_output, z_so_far)

        # Concatenate with previously generated latent dims
        conc = Concatenate(axis=1, name='concat')
        mean_so_far = conc([mean_so_far, mean_k])
        logvar_so_far = conc([logvar_so_far, logvar_k])
        z_so_far = conc([z_so_far, z_k])

        return [mean_so_far, logvar_so_far, z_so_far, src_seq, enc_output]

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask


class InterimDecoder2():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, decoder_width=1, dropout=0.1, word_emb=None, pos_emb=None, latent_dim=None, stddev=1,
                 false_emb=None):
        self.latent_dim = latent_dim
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.d_model = d_model
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]
        self.width = decoder_width

        # Calculate 'decoder_width' means/variances from output
        # self.mean_layer = TimeDistributed(Dense(1, input_shape=(d_model,)), name='ID2_mean_layer')
        # self.logvar_layer = TimeDistributed(Dense(1, input_shape=(d_model,)), name='ID2_logvar_layer')
        self.mean_layer = Dense(self.latent_dim, input_shape=(self.latent_dim * d_model,), name='ID2_mean_layer')
        self.logvar_layer = Dense(self.latent_dim, input_shape=(self.latent_dim * d_model,), name='ID2_logvar_layer')
        self.mean_layer2 = Dense(self.latent_dim, input_shape=(self.latent_dim,), name='ID2_mean_layer2')
        self.logvar_layer2 = Dense(self.latent_dim, input_shape=(self.latent_dim,), name='ID2_logvar_layer2')
        # For the very first vector
        self.attn = TimeDistributed(Dense(self.width, input_shape=(d_model,), activation='linear'), name='ID2_ATTN')

        self.squeeze = Lambda(lambda x: K.squeeze(x, axis=-1), name='ID2_SQUEEZE')

        self.ldim = K.constant(latent_dim, dtype='int32')
        self.concat = Concatenate(axis=1, name='ID2_CONCAT')

    def __call__(self, src_seq, enc_output):
        '''
        Will iteratively compute means and variances
        :param src_seq:
        :param enc_output:
        :return:
        '''
        z_init = self.first_iter(src_seq, enc_output)

        def the_loop(args):
            z_init_ = args

            # use interim decoder to generate latent dimension iteratively
            z_embedded, _, _ = tf.while_loop(self.cond, self.step,
                                             [z_init_, src_seq, enc_output],
                                             shape_invariants=[tf.TensorShape([None, None, self.d_model]),
                                                               src_seq.get_shape(),
                                                               enc_output.get_shape()], name='ID2_LOOP')

            # will generate too many latent dimensions, so clip it
            z_embedded = z_embedded[:, :self.latent_dim, :]
            return K.reshape(z_embedded, [-1, self.latent_dim * self.d_model])

        z_emb = Lambda(the_loop)(z_init)
        means = self.mean_layer(z_emb)
        means = self.mean_layer2(means)
        logvars = self.logvar_layer(z_emb)
        logvars = self.logvar_layer2(logvars)
        return means, logvars

    def compute_next_z(self, z_so_far, src_seq, enc_output):
        # z is [batch_size, k]
        # Embed to be of size [batch_size, k, d_model]
        z = z_so_far

        # Positional embedding
        # z_pos = Lambda(self.get_pos_seq)(z_so_far)
        # z_pos = self.pos_layer(z_pos)
        # z = Add()([z, z_pos])

        # Mask the encoder output
        self_mask = None
        enc_mask = None
        # enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([z_so_far, src_seq])
        # Run through interim decoder
        for dec_layer in self.layers:
            z, self_att, enc_att = dec_layer(z, enc_output, self_mask, enc_mask)

        return z

    def step(self, z_so_far, src_seq, enc_output):
        '''
        Given latent values generated so far, will generate next decoder_width
        latent means and variances
        :param src_seq:     Embedded source sequence
        :param enc_output:  Output of encoder
        :param z_so_far:    Sampled latent variables generated so far
        :return:
        '''

        z_i = self.compute_next_z(z_so_far, src_seq, enc_output)
        s = Lambda(lambda x: x[:, -self.width:, :], name='ID2_LOOPSTRIDE')
        s2 = Lambda(lambda x: K.reshape(x, [-1, self.width, self.d_model]), name='ID2_LOOPSHAPE')
        z_i = s2(s(z_i))
        z = self.concat([z_so_far, z_i])

        return [z, src_seq, enc_output]

    def first_iter(self, src_seq, enc_output, return_att=False, active_layers=999):
        # We generate the first [width] vectors using a weighted average of the encoder output
        a_vals = self.attn(enc_output)
        a_vals = Softmax(axis=1)(a_vals)
        h = Dot(axes=1)([a_vals, enc_output])

        # h is now [batch_size, width, d_model]
        # The code in "step" must be run at least once
        # Outside the loop just to make Keras happy
        h = self.compute_next_z(h, src_seq, enc_output)

        return h

    def cond(self, z_so_far, src_seq, enc_output):
        # Return true while mean length is less than latent dim
        return tf.less(K.shape(z_so_far)[1], self.ldim)

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask


class InterimDecoder3():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, decoder_width=1, dropout=0.1, word_emb=None, pos_emb=None, latent_dim=None, stddev=1,
                 false_emb=None):
        self.latent_dim = latent_dim
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.d_model = d_model
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]
        self.width = decoder_width

        # Calculate means/variances from output
        fin_length = np.ceil(self.latent_dim / decoder_width)
        self.mean_layer = Dense(self.latent_dim, input_shape=(d_model * fin_length,), name='ID3_mean_layer')
        self.logvar_layer = Dense(self.latent_dim, input_shape=(d_model * fin_length,), name='ID3_logvar_layer')
        self.query_layer = TimeDistributed(Dense(self.d_model, input_shape=(self.d_model,), name='ID3_Query'))
        self.value_layer = TimeDistributed(Dense(self.d_model, input_shape=(self.d_model,), name='ID3_Values'))
        # For the very first vector
        self.attn = TimeDistributed(Dense(1, input_shape=(d_model,), activation='linear'), name='ID3_ATTN')

        self.squeeze = Lambda(lambda x: K.squeeze(x, axis=-1), name='ID2_SQUEEZE')

        self.ldim = K.constant(fin_length, dtype='int32')
        self.next_z = Lambda(self.compute_next_z, name='ComputeZ')

    def __call__(self, src_seq, enc_output):
        '''
        Will iteratively compute means and variances
        :param src_seq:
        :param enc_output:
        :return:
        '''
        z_init = self.first_iter(src_seq, enc_output)

        def the_loop(args):
            z_init_ = args

            # use interim decoder to generate latent dimension iteratively
            z_embedded, _, _ = tf.while_loop(self.cond, self.step,
                                             [z_init_, src_seq, enc_output],
                                             shape_invariants=[tf.TensorShape([None, None, self.d_model]),
                                                               src_seq.get_shape(),
                                                               enc_output.get_shape()], name='ID2_LOOP')

            # will generate too many latent dimensions, so clip it
            return K.reshape(z_embedded, [-1, self.ldim * self.d_model])
            # return K.reshape(z_embedded, )

        z_emb = Lambda(the_loop)(z_init)

        return self.mean_layer(z_emb), self.logvar_layer(z_emb)

    def compute_next_z(self, args):
        z_so_far, src_seq, enc_output = args
        z = z_so_far

        # # Positional embedding
        # z_pos = Lambda(self.get_pos_seq)(z_so_far)
        # z_pos = self.pos_layer(z_pos)
        # z = Add()([z, z_pos])

        # Mask the output
        # self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(z_so_far)
        # self_sub_mask = Lambda(GetSubMask)(z_so_far)
        # self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])

        # if return_att: self_atts, enc_atts = [], []
        # self_mask = None

        # enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([z_so_far, src_seq])

        # Run through interim decoder
        for dec_layer in self.layers:
            z, self_att, enc_att = dec_layer(z, enc_output)  # , self_mask, enc_mask)
            # if return_att:
            #     self_atts.append(self_att)
            #     enc_atts.append(enc_att)

        # key = Lambda(lambda x: K.expand_dims(x[:, -1, :], axis=1))(z)
        key = K.expand_dims(z[:, -1, :], axis=1)
        # key = z[:, -1, :]
        queries = self.query_layer(enc_output)
        values = self.value_layer(enc_output)
        # Attention values
        a = Dot(axes=2)([queries, key])
        a = Softmax(axis=1)(a / np.sqrt(self.d_model))  # [bs x length x 1]
        z_hat = Dot(axes=1)([values, a])  # [bs x dmodel]
        z_hat = K.reshape(z_hat, [-1, 1, self.d_model])
        # z_hat = Lambda(lambda x: K.reshape(x, [-1, 1, self.d_model]))(z_hat)
        # Decoder output is also [batch_size, k, d_model]
        # new_z = z[:, -self.width:, :self.d_model]
        return z_hat

    def step(self, z_so_far, src_seq, enc_output):
        '''
        Given latent values generated so far, will generate next decoder_width
        latent means and variances
        :param src_seq:     Embedded source sequence
        :param enc_output:  Output of encoder
        :param z_so_far:    Sampled latent variables generated so far
        :return:
        '''
        # z_i = Lambda(self.compute_next_z)
        z_i = self.next_z([z_so_far, src_seq, enc_output])
        z = Concatenate(axis=1)([z_so_far, z_i])

        return [z, src_seq, enc_output]

    def first_iter(self, src_seq, enc_output, return_att=False, active_layers=999):
        # We generate the first [width] vectors using a weighted average of the encoder output
        a_vals = self.attn(enc_output)
        a_vals = Softmax(axis=1)(a_vals)
        h = Dot(axes=1)([a_vals, enc_output])

        # h is now [batch_size, width, d_model]
        # The code in "step" must be run at least once
        # Outside the loop just to make Keras happy
        h = self.next_z([h, src_seq, enc_output])

        return h

    def cond(self, z_so_far, src_seq, enc_output):
        # Return true while mean length is less than latent dim
        return tf.less(K.shape(z_so_far)[1], self.ldim)

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask


class InterimDecoder4():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, decoder_width=1, dropout=0.1, word_emb=None, pos_emb=None, latent_dim=None, stddev=1,
                 false_emb=None):
        self.latent_dim = latent_dim
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.d_model = d_model
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

        # Calculate 'decoder_width' means/variances from output
        dim1 = int(2 ** np.ceil(np.log2(self.latent_dim)))
        self.mean_layer = Dense(self.latent_dim, input_shape=(dim1 * d_model,), activation='relu',
                                name='ID2_mean_layer')
        self.logvar_layer = Dense(self.latent_dim, input_shape=(dim1 * d_model,), activation='relu',
                                  name='ID2_logvar_layer')
        self.mean_layer2 = Dense(self.latent_dim, input_shape=(self.latent_dim,), name='ID2_mean_layer2')
        self.logvar_layer2 = Dense(self.latent_dim, input_shape=(self.latent_dim,), name='ID2_logvar_layer2')

        # For the very first vector
        self.init_mode = "attention"
        if self.init_mode == "attention":
            self.attn = TimeDistributed(Dense(1, input_shape=(d_model,), activation='linear'), name='ID2_ATTN')
            self.vals = TimeDistributed(Dense(d_model, input_shape=(d_model,)))
            self.squeeze = Lambda(lambda x: K.squeeze(x, axis=-1), name='ID2_SQUEEZE')

        self.ldim = K.constant(latent_dim, dtype='int32')
        self.concat = Concatenate(axis=1, name='ID2_CONCAT')

    def __call__(self, src_seq, enc_output):
        '''
        Will iteratively compute means and variances
        :param src_seq:
        :param enc_output:
        :return:
        '''
        z_init = self.first_iter(src_seq, enc_output)

        def the_loop(args):
            z_init_ = args

            # use interim decoder to generate latent dimension iteratively
            z_embedded, _, _ = tf.while_loop(self.cond, self.step,
                                             [z_init_, src_seq, enc_output],
                                             shape_invariants=[tf.TensorShape([None, None, self.d_model]),
                                                               src_seq.get_shape(),
                                                               enc_output.get_shape()], name='ID2_LOOP')

            # will generate too many latent dimensions, so clip it
            z_embedded = self.compute_next_z(z_embedded, src_seq, enc_output)
            dim1 = int(2 ** np.ceil(np.log2(self.latent_dim)))
            return K.reshape(z_embedded, [-1, dim1 * self.d_model])

        z_emb = Lambda(the_loop)(z_init)
        means = self.mean_layer(z_emb)
        means = self.mean_layer2(means)
        logvars = self.logvar_layer(z_emb)
        logvars = self.logvar_layer2(logvars)
        return means, logvars

    def compute_next_z(self, z_so_far, src_seq, enc_output):
        # z is [batch_size, k]
        # Embed to be of size [batch_size, k, d_model]
        z = z_so_far

        # Positional embedding
        # z_pos = Lambda(self.get_pos_seq)(z_so_far)
        # z_pos = self.pos_layer(z_pos)
        # z = Add()([z, z_pos])

        # Mask the encoder output
        self_mask = None
        enc_mask = None
        # enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([z_so_far, src_seq])
        # Run through interim decoder
        for dec_layer in self.layers:
            z, self_att, enc_att = dec_layer(z, enc_output, self_mask, enc_mask)

        return z

    def step(self, z_so_far, src_seq, enc_output):
        '''
        Given latent values generated so far, will generate next decoder_width
        latent means and variances
        :param src_seq:     Embedded source sequence
        :param enc_output:  Output of encoder
        :param z_so_far:    Sampled latent variables generated so far
        :return:
        '''

        z_i = self.compute_next_z(z_so_far, src_seq, enc_output)
        z = self.concat([z_so_far, z_i])

        return [z, src_seq, enc_output]

    def first_iter(self, src_seq, enc_output, return_att=False, active_layers=999):
        # We generate the first [width] vectors using a weighted average of the encoder output
        if self.init_mode == "attention":
            a_vals = self.attn(enc_output)
            a_vals = Softmax(axis=1)(a_vals)
            z = Dot(axes=1)([a_vals, self.vals(enc_output)])
        elif self.init_mode == "zeros":
            def get_zeros(arg):
                bs = K.shape(arg)[0]
                return K.zeros([bs, 1, self.d_model], dtype='float')

            z = Lambda(get_zeros)(src_seq)
        elif self.init_mode == "random":
            def get_zeros(arg):
                bs = K.shape(arg)[0]
                return K.random_normal([bs, 1, self.d_model], stddev=0.01, dtype='float')

            z = Lambda(get_zeros)(src_seq)

        # z is now [batch_size, 1, d_model]
        # The code in "step" must be run at least once
        # Outside the loop just to make Keras happy
        return self.compute_next_z(z, src_seq, enc_output)

    def cond(self, z_so_far, src_seq, enc_output):
        # Return true while mean length is less than latent dim
        return tf.less(K.shape(z_so_far)[1], self.ldim)

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask


class InterimDecoder5():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, decoder_width=1, dropout=0.1, word_emb=None, pos_emb=None, latent_dim=None, stddev=1,
                 false_emb=None):
        self.latent_dim = latent_dim
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.d_model = d_model
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

        # Calculate 'decoder_width' means/variances from output
        self.len = int(np.ceil(latent_dim / decoder_width))
        self.attn = KQV_Attn(d_in=d_model, d_out=d_model)
        self.mean_layer = Dense(self.latent_dim, input_shape=(self.len * d_model,), activation='relu',
                                name='ID2_mean_layer')
        self.logvar_layer = Dense(self.latent_dim, input_shape=(self.len * d_model,), activation='relu',
                                  name='ID2_logvar_layer')
        self.mean_layer2 = Dense(self.latent_dim, input_shape=(self.latent_dim,), name='ID2_mean_layer2')
        self.logvar_layer2 = Dense(self.latent_dim, input_shape=(self.latent_dim,), name='ID2_logvar_layer2')

        self.ldim = K.constant(latent_dim, dtype='int32')
        self.concat = Concatenate(axis=1, name='ID2_CONCAT')

    def __call__(self, src_seq, enc_output):
        '''
        Will iteratively compute means and variances
        :param src_seq:
        :param enc_output:
        :return:
        '''
        z_init = self.first_iter(src_seq, enc_output)

        def the_loop(args):
            z_init_ = args

            # use interim decoder to generate latent dimension iteratively
            z_embedded, _, _ = tf.while_loop(self.cond, self.step,
                                             [z_init_, src_seq, enc_output],
                                             shape_invariants=[tf.TensorShape([None, None, self.d_model]),
                                                               src_seq.get_shape(),
                                                               enc_output.get_shape()], name='ID2_LOOP')

            # flatten the vector
            # return K.batch_flatten(z_embedded)
            return K.reshape(z_embedded, [-1, int(self.len * self.d_model)])

        z_emb = Lambda(the_loop)(z_init)
        means = self.mean_layer(z_emb)
        means = self.mean_layer2(means)
        logvars = self.logvar_layer(z_emb)
        logvars = self.logvar_layer2(logvars)
        return means, logvars

    def compute_next_z(self, z_so_far, src_seq, enc_output):
        # z is [batch_size, k]
        # Embed to be of size [batch_size, k, d_model]
        z = z_so_far

        # Positional embedding
        # z_pos = Lambda(self.get_pos_seq)(z_so_far)
        # z_pos = self.pos_layer(z_pos)
        # z = Add()([z, z_pos])

        # Mask the encoder output
        self_mask = None
        enc_mask = None
        # enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([z_so_far, src_seq])
        # Run through interim decoder
        for dec_layer in self.layers:
            z, self_att, enc_att = dec_layer(z, enc_output, self_mask, enc_mask)

        return self.attn(z)

    def step(self, z_so_far, src_seq, enc_output):
        '''
        Given latent values generated so far, will generate next decoder_width
        latent means and variances
        :param src_seq:     Embedded source sequence
        :param enc_output:  Output of encoder
        :param z_so_far:    Sampled latent variables generated so far
        :return:
        '''

        z_i = self.compute_next_z(z_so_far, src_seq, enc_output)

        z_i = Lambda(lambda x: K.expand_dims(x, axis=1))(z_i)
        z = self.concat([z_so_far, z_i])

        return [z, src_seq, enc_output]

    def first_iter(self, src_seq, enc_output, return_att=False, active_layers=999):
        # Generate first latent vector using vector of zeros
        def get_zeros(arg):
            bs = K.shape(arg)[0]
            return K.zeros([bs, 1, self.d_model], dtype='float')

        # z is now [batch_size, 1, d_model]
        z = Lambda(get_zeros)(src_seq)

        # The code in "step" must be run at least once
        # Outside the loop just to make Keras happy
        z = self.compute_next_z(z, src_seq, enc_output)
        z = Lambda(lambda x: K.expand_dims(x, axis=1))(z)
        return z

    def cond(self, z_so_far, src_seq, enc_output):
        # Return true while mean length is less than latent dim
        return tf.less(K.shape(z_so_far)[1], self.len)

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask


class FalseEmbeddingsTD():
    def __init__(self, d_emb, d_latent=None, residual=True, layers=1):
        '''
        Given a 1D vector, attempts to create 'false' embeddings to
        go from latent space
        :param d_emb: dimensionality of false embeddings
        '''
        self.l1 = Dense(d_latent, input_shape=(d_latent,), activation='relu')
        self.l2 = Dense(d_latent, input_shape=(d_latent,))
        self.init_layer = TimeDistributed(Dense(d_emb, input_shape=(1,)))
        self.deep_layers = [TimeDistributed(Dense(d_emb, activation='relu', input_shape=(d_emb,))) for _ in
                            range(layers)]

        # Whether or not to employ residual connection
        self.residual = residual
        if self.residual:
            self.deep_res_layers = [TimeDistributed(Dense(d_emb, input_shape=(d_emb,))) for _ in
                                    range(layers)]
        self.final_lin = TimeDistributed(Dense(d_emb, input_shape=(d_emb,)))

    def __call__(self, z):
        '''
        :param z: Input with dimensionality [batch_size, d] or [batch_size, d, 1]
        :return: Falsely embedded output with dimensionality [batch_size, d, d_emb]
        '''
        z = self.l1(z)
        z = self.l2(z)
        if z.shape.ndims == 2:
            z = Lambda(lambda x: K.expand_dims(x, axis=2))(z)

        # use fully connected layer to expand to [batch_size, d, d_emb]
        z = self.init_layer(z)

        for (i, layer) in enumerate(self.deep_layers):
            if self.residual:
                z_w = self.deep_res_layers[i](layer(z))
                z = Add()([z, z_w])
                # z = z + layer(z)
            else:
                z = layer(z)

        return self.final_lin(z)


class FalseEmbeddings():
    def __init__(self, d_emb, d_latent, residual=True, layers=0):
        '''
        Given a 1D vector, attempts to create 'false' embeddings to
        go from latent space
        :param d_emb: dimensionality of false embeddings
        '''

        latent_len = int(np.ceil(d_latent / d_emb))
        self.init_layer = Dense(d_emb * latent_len, input_shape=(d_latent,))

        # self.init_layer = TimeDistributed(Dense(d_emb, input_shape=(1,)))
        self.deep_layers = [Dense(d_latent, activation='relu', input_shape=(d_latent,)) for _ in
                            range(layers)]

        self.deep_time_layers = [TimeDistributed(Dense(d_emb, activation='relu', input_shape=(d_emb,))) for _ in
                                 range(layers)]

        # Whether or not to employ residual connection
        self.residual = residual
        if self.residual:
            self.deep_res_layers = [Dense(d_latent, input_shape=(d_latent,)) for _ in
                                    range(layers)]

            self.deep_res_time_layers = [TimeDistributed(Dense(d_emb, input_shape=(d_emb,))) for _ in
                                         range(layers)]

        # self.time_norm = TimeDistributed(BatchNormalization())
        self.final_shape = Lambda(lambda x: K.reshape(x, [-1, latent_len, d_emb]))

        # Add positional embedding?
        train_posemb = True
        if train_posemb:
            self.pos_emb = Embedding(latent_len, d_emb, trainable=True)
        else:
            self.pos_emb = Embedding(latent_len, d_emb, trainable=False,
                                     weights=[GetPosEncodingMatrix(latent_len, d_emb)])

        self.pos_seq = Lambda(lambda x: self.pos_emb(K.cumsum(K.ones([K.shape(x)[0], latent_len], 'int32'), 1) - 1))

        # self.pos_emb = None

    def __call__(self, z):
        '''

        :param z: Input with dimensionality [batch_size, d] or [batch_size, d, 1]
        :return: Falsely embedded output with dimensionality [batch_size, d, d_emb]
        '''

        # use fully connected layer to expand to [batch_size, d, d_emb]

        for (i, layer) in enumerate(self.deep_layers):
            if self.residual:
                # z = ReLU((WReLU(Wz) + z))
                z_w = self.deep_res_layers[i](layer(z))
                z = Add()([z, z_w])
                z = BatchNormalization()(z)
                z = ReLU()(z)
                # z = z + layer(z)
            else:
                z = layer(z)
                z = BatchNormalization()(z)

        z = self.init_layer(z)
        z = self.final_shape(z)

        if self.pos_emb:
            z = Add()([z, self.pos_seq(z)])

        for (i, layer) in enumerate(self.deep_time_layers):
            if self.residual:
                z_w = self.deep_res_time_layers[i](layer(z))
                z = Add()([z, z_w])
                z = TimeDistributed(BatchNormalization)(z)
                z = ReLU()(z)
                # z = z + layer(z)
            else:
                z = layer(z)
                z = TimeDistributed(BatchNormalization)(z)

        return z


class TransformerDecoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output, return_att=False, active_layers=999):

        dec = self.emb_layer(tgt_seq)
        pos = self.pos_layer(tgt_pos)
        x = Add()([dec, pos])

        self_pad_mask = Lambda(lambda x: GetPadMask(x, x), name='DecoderPadMask')(tgt_seq)
        self_sub_mask = Lambda(GetSubMask, name='DecoderSubMask')(tgt_seq)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]), name='DecoderSelfMask')([self_pad_mask, self_sub_mask])

        # Don't want encoder mask as there should be no padding from latent dim
        enc_mask = None
        # Lambda(lambda x: GetPadMask(x[0], x[1]), name='DecoderEncMask')([tgt_seq, src_seq])

        if return_att: self_atts, enc_atts = [], []

        for dec_layer in self.layers[:active_layers]:
            x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)

            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)

        return (x, self_atts, enc_atts) if return_att else x


class FiLM():
    def __init__(self, d_latent, d_emb):
        self.token_predense = TimeDistributed(Dense(d_emb, activation='relu'))
        self.latent_predense = Dense(d_latent, activation='relu')
        self.concat_predense = TimeDistributed(Dense(d_emb, activation='relu'))
        self.gamma = TimeDistributed(Dense(d_emb))
        self.beta = TimeDistributed(Dense(d_emb))

    def __call__(self, h, z):
        '''

        :param h: decoder representation
        :param z: latent vector
        :return: FiLMed decoder representation
        '''
        z = self.latent_predense(z)
        h_ = self.token_predense(h)
        z = Lambda(lambda x: K.repeat(x, K.shape(h)[1]))(z)
        zh = Concatenate(axis=2)([z, h_])
        zh = self.concat_predense(zh)
        gamma = self.gamma(zh)
        beta = self.beta(zh)
        return Lambda(lambda x: x[0]*zh + x[1])([gamma, beta])

class DecoderLayerFiLM():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, d_latent, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        self.film_layer = FiLM(d_latent, d_model)

    def __call__(self, dec_input, enc_output, z_latent, self_mask=None, enc_mask=None):
        output, slf_attn = self.self_att_layer(dec_input, dec_input, dec_input, mask=self_mask)
        output, enc_attn = self.enc_att_layer(output, enc_output, enc_output, mask=enc_mask)
        output = self.film_layer(output, z_latent)
        output = self.pos_ffn_layer(output)
        return output, slf_attn, enc_attn


class DecoderWithFILM():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, d_latent,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayerFiLM(d_model, d_inner_hid, n_head, d_k, d_v, d_latent, dropout) for _ in range(layers)]

    def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output, z_latent, return_att=False, active_layers=999):

        dec = self.emb_layer(tgt_seq)
        pos = self.pos_layer(tgt_pos)
        x = Add()([dec, pos])

        self_pad_mask = Lambda(lambda x: GetPadMask(x, x), name='DecoderPadMask')(tgt_seq)
        self_sub_mask = Lambda(GetSubMask, name='DecoderSubMask')(tgt_seq)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]), name='DecoderSelfMask')([self_pad_mask, self_sub_mask])

        # Don't want encoder mask as there should be no padding from latent dim
        enc_mask = None
        # Lambda(lambda x: GetPadMask(x[0], x[1]), name='DecoderEncMask')([tgt_seq, src_seq])

        if return_att: self_atts, enc_atts = [], []

        for dec_layer in self.layers[:active_layers]:
            x, self_att, enc_att = dec_layer(x, enc_output, z_latent, self_mask, enc_mask)

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
        self.basic = (d_model ** -0.5)
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

latent_dict = {"average1": AvgLatent,
               "average2": AvgLatent2,
               "average3": AvgLatent3,
               "average4": AvgLatent4,
               "sum1": SumLatent,
               "sum2": SumLatent2,
               "ar1": InterimDecoder,
               "ar2": InterimDecoder2,
               "ar3": InterimDecoder3,
               "ar_log": InterimDecoder4,
               "ar_slim": InterimDecoder5}
