import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import objectives
from keras.layers import Input, Lambda, Dot
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Flatten, RepeatVector
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils.training_utils import multi_gpu_model

import molecules.transformer as tr
from molecules.transformer import debugPrint
from utils import DefaultDecoderParams


class MoleculeVAE():
    autoencoder = None

    def __init__(self, tokens, params):
        self.p = params
        # self.p["len_limit"] = p["len_limit"]
        self.charset_length = tokens.num()
        x = Input(shape=(self.p["len_limit"], self.charset_length))
        _, z, mean, logvar = self._buildEncoder(x)
        self.encode_sample = Model(x, z)
        self.encode = Model(x, [mean, logvar])

        encoded_input = Input(shape=(self.p["latent_dim"],))
        self.decode = Model(
            encoded_input,
            self._buildDecoder(encoded_input)
        )

        x1 = Input(shape=(self.p["len_limit"], self.charset_length))
        vae_loss, z1, mean, logvar = self._buildEncoder(x1)
        p1 = self._buildPropertyPredictor(z1)
        self.autoencoder = Model(
            x1,
            [self._buildDecoder(z1), p1]
        )

        self.autoencoder.compile(optimizer='Adam',
                                 loss=[vae_loss, 'mean_squared_error'],
                                 loss_weights=[1.0, self.p["pp_weight"] / self.p["num_props"] ** 2],
                                 metrics=['accuracy'])

    def _buildEncoder(self, x):
        h = Convolution1D(9, 9, activation='relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation='relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, self.p["latent_dim"]), mean=0., stddev=self.p["stddev"])
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(self.p["latent_dim"], name='z_mean', activation='linear')(h)
        z_log_var = Dense(self.p["latent_dim"], name='z_log_var', activation='linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = self.p["len_limit"] * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return vae_loss, Lambda(sampling, output_shape=(self.p["latent_dim"],), name='lambda')(
            [z_mean, z_log_var]), z_mean, z_log_var

    def _buildDecoder(self, z):
        h = Dense(self.p["latent_dim"], name='latent_input', activation='relu')(z)
        h = RepeatVector(self.p["len_limit"], name='repeat_vector')(h)
        h = GRU(501, return_sequences=True, name='gru_1')(h)
        h = GRU(501, return_sequences=True, name='gru_2')(h)
        h = GRU(501, return_sequences=True, name='gru_3')(h)

        return TimeDistributed(Dense(self.charset_length, activation='softmax'), name='decoder')(h)

    def _buildPropertyPredictor(self, x):
        h = Dense(self.p["latent_dim"], input_shape=(self.p["latent_dim"],), activation='linear')(x)
        for _ in range(self.p["pp_layers"] - 1):
            h = Dense(self.p["pp_layers"], activation='linear')(h)
        return Dense(self.p["num_props"], activation='linear', name='props')(h)


# https://github.com/Lsdefine/attention-is-all-you-need-keras
class TriTransformer:
    def __init__(self, i_tokens, params, o_tokens=None):
        self.i_tokens = i_tokens
        params["pp_weight"] = 0 if params["bottleneck"] == "none" else params["pp_weight"]
        self.p = params
        self.o_tokens = i_tokens  # Autoencoder

        pos_emb = tr.Embedding(params["len_limit"], params["d_model"], trainable=False,
                               weights=[tr.GetPosEncodingMatrix(params["len_limit"], params["d_model"])])
        self.word_emb = tr.Embedding(self.o_tokens.num(), self.p["d_model"])
        self.decode = None
        if self.p["bottleneck"] == "conv" or "gru" in self.p["bottleneck"]:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(params=params, tokens=i_tokens,
                                              pos_emb=pos_emb, word_emb=self.word_emb)
        # self.encoder =
        self.kl_loss_var = K.variable(0.0, dtype=np.float, name='kl_loss_weight') if self.p["stddev"] else None

        # Sample from latent space
        def sampling(args):
            z_mean_, z_logvar_ = args
            if not self.p["stddev"]:
                return z_mean_
            else:
                if self.p["bottleneck"] == "none":
                    latent_shape = (K.shape(z_mean_)[0], K.shape(z_mean_)[1], self.p["d_model"])
                else:
                    latent_shape = (K.shape(z_mean_)[0], self.p["latent_dim"])

                epsilon = K.random_normal(shape=latent_shape, mean=0.,
                                          stddev=self.p["stddev"] * self.kl_loss_var / self.p["kl_max_weight"])
                return z_mean_ + K.exp(z_logvar_ / 2) * epsilon

        self.sampler = Lambda(sampling)
        use_default_decoder = True
        if use_default_decoder:
            self.p = DefaultDecoderParams()
            dec_pos_emb = tr.Embedding(self.p["len_limit"], self.p["d_model"], trainable=False,
                                       weights=[tr.GetPosEncodingMatrix(params["len_limit"], self.p["d_model"])])
            dec_word_emb = tr.Embedding(self.o_tokens.num(), self.p["d_model"])
        else:
            dec_word_emb = self.word_emb
            dec_pos_emb = pos_emb

        self.false_embedder = tr.FalseEmbeddings(d_emb=self.p["d_model"], d_latent=self.p["latent_dim"])

        self.decoder = tr.DecoderFromLatent(self.p["d_model"], self.p["d_inner_hid"], self.p["heads"], self.p["d_k"],
                                            self.p["d_v"], self.p["layers"], self.p["dropout"],
                                            word_emb=dec_word_emb, pos_emb=dec_pos_emb)
        self.p = params
        self.target_layer = TimeDistributed(Dense(self.o_tokens.num(), use_bias=False))

        self.metrics = {}

    def _buildGRUEncoder(self, x, no_attn=False):
        h = self.word_emb(x)
        # For now, avoid having to introduce new commandline parameters
        for _ in range(self.p["ID_layers"]):
            h = GRU(self.p["ID_d_k"], return_sequences=True)(h)
        if no_attn:
            h = h[:, -1, :]
        else:
            values = TimeDistributed(Dense(self.p["latent_dim"]))(h)
            queries = TimeDistributed(Dense(self.p["latent_dim"]))(h)
            keys = TimeDistributed(Dense(self.p["latent_dim"]))(h)

            attn_vals = Dot(axes=2)([keys, queries])
            attn_vals = Lambda(lambda x: K.sum(x, axis=2, keepdims=True))(attn_vals)

            h = Dot(axes=1)([attn_vals, values])
            h = Lambda(lambda x: K.squeeze(x, axis=1))(h)
        z_mean = Dense(self.p["latent_dim"], name='z_mean', activation='linear')(h)
        z_log_var = Dense(self.p["latent_dim"], name='z_log_var', activation='linear')(h)
        return z_mean, z_log_var

    def _buildConvEncoder(self, x):
        h = Convolution1D(9, 9, activation='relu')(x)
        h = Convolution1D(9, 9, activation='relu')(h)
        h = Convolution1D(10, 11, activation='relu')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu')(h)
        z_mean = Dense(self.p["latent_dim"], name='z_mean', activation='linear')(h)
        z_log_var = Dense(self.p["latent_dim"], name='z_log_var', activation='linear')(h)
        return z_mean, z_log_var

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def build_models(self, active_layers=999):
        tgt_seq_in = Input(shape=(None,), dtype='int32', name='tgt_seq')
        tgt_seq = Lambda(lambda x: x[:, :-1])(tgt_seq_in)
        tgt_true = Lambda(lambda x: x[:, 1:])(tgt_seq_in)
        tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)

        if self.encoder is None:
            enc_attn = None
            if "gru" in self.p["bottleneck"]:
                src_seq = Input(shape=(None,), dtype='int32', name='src_seq')
                z_mean, z_logvar = self._buildGRUEncoder(src_seq, no_attn=("na" in self.p["bottleneck"]))
            else:
                src_seq = Input(shape=(self.p["len_limit"], self.i_tokens.num()), name='src_seq')
                z_mean, z_logvar = self._buildConvEncoder(src_seq)
            z_sampled = None
        else:
            src_seq = Input(shape=(None,), dtype='int32', name='src_seq')
            src_pos = Lambda(self.get_pos_seq)(src_seq)
            z_mean, z_logvar, z_sampled, enc_attn = self.encoder(src_seq, src_pos)

        # Sample
        if z_sampled is None:
            z_sampled = self.sampler([z_mean, z_logvar])

        # generate an 'input' sampled value so we can create a separate
        # model to decode from latent space
        if self.p["bottleneck"] == "none":
            z_input = Input(shape=(None, self.p["d_model"]), dtype='float32', name='z_input')
        else:
            z_input = Input(shape=(self.p["latent_dim"],), dtype='float32', name='z_input')

        # must calculate for both a latent input and the full end-to-end system
        final_output = []
        props = []
        for l_vec in [z_input, z_sampled]:
            # 'false embed' for decoder
            dec_input = self.false_embedder(l_vec)
            dec_output, dec_attn, encdec_attn = self.decoder(tgt_seq,
                                                             tgt_pos,
                                                             l_vec,
                                                             dec_input,
                                                             active_layers=active_layers,
                                                             return_att=True)

            dec_output = debugPrint(dec_output, "DEC_OUTPUT")
            # Do not perform softmax on output
            # As it is performed in the loss function
            final_output.append(self.target_layer(dec_output))

            # Property prediction
            if self.p["pp_weight"] is not None:
                props.append(self._buildPropertyPredictor(l_vec))

        # KL DIVERGENCE LOSS
        def kl_loss(args):
            z_mean_, z_log_var_ = args

            return - 0.5 * self.kl_loss_var * tf.reduce_mean(
                1 + z_log_var_ - K.square(z_mean_) - K.exp(z_log_var_),
                name='KL_loss_sum')

        losses = []

        # RECONSTRUCTION LOSS
        def get_loss(args):
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            loss = K.mean(loss)
            return loss

        def get_acc(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        losses.append(Lambda(get_loss, name='Loss')([final_output[1], tgt_true]))
        self.metrics["ppl"] = Lambda(K.exp)(losses[0])

        # VARIATIONAL LOSS
        if self.kl_loss_var is not None:
            if self.p["WAE_s"] == 0 or self.p["WAE_kernel"] is None:
                print("Using variational autoencoder")
                kl = Lambda(kl_loss, name='VariationalLoss')([z_mean, z_logvar])
            else:
                kl = Lambda(self.mmd_penalty, name='WAELoss')(z_sampled)
                self.metrics["wae"] = Lambda(self.wae_mmd_exact, name='VariationalLoss')(
                    [z_mean, z_logvar])
            kl = Lambda(lambda x: self.kl_loss_var * x)(kl)
            self.metrics["kl_loss"] = kl
            losses.append(kl)

        if self.p["pp_weight"]:
            prop_input = Input(shape=[self.p["num_props"]])
            pp_loss = Lambda(lambda x: tf.losses.mean_squared_error(x, props[1],
                                                                    weights=self.p["pp_weight"] / (
                                                                            self.p["num_props"] ** 2)))(prop_input)
            self.metrics["pp_loss"] = pp_loss
            losses.append(pp_loss)

        loss = Lambda(tf.reduce_sum)(losses)

        # Set up autoencoder model
        if self.p["pp_weight"] is None:
            self.autoencoder = Model([src_seq, tgt_seq_in], loss)
        else:
            self.autoencoder = Model([src_seq, tgt_seq_in, prop_input], loss)

        self.autoencoder.add_loss([loss])

        ## METRICS
        self.metrics["acc"] = Lambda(get_acc)([final_output[1], tgt_true])
        self.metrics["z_mean"] = Lambda(tf.reduce_mean)(z_mean)
        if self.kl_loss_var is not None: self.metrics["z_logvar"] = Lambda(tf.reduce_mean)(z_logvar)

        # For outputting next symbol
        self.output_model = Model([src_seq, tgt_seq_in], final_output[1])

        # For getting attentions
        attn_list = dec_attn + encdec_attn if enc_attn is None else enc_attn + dec_attn + encdec_attn
        self.output_attns = Model([src_seq, tgt_seq_in], attn_list)

        # ENCODING/DECODING MODELS
        self.encode = Model(src_seq, [z_mean, z_logvar])
        self.encode_sample = Model(src_seq, [z_sampled])
        self.decode = Model([z_input, tgt_seq_in], final_output[0])

    def _buildPropertyPredictor(self, x):
        h = Dense(self.p["latent_dim"], input_shape=(self.p["latent_dim"],), activation='relu')(x)
        for _ in range(self.p["pp_layers"] - 1):
            h = Dense(self.p["pp_layers"], activation='relu')(h)
        return Dense(self.p["num_props"], activation='linear', name='props')(h)

    def compile_vae(self, optimizer='adam', N_GPUS=1):
        self.encode_sample.compile('adam', 'mse')
        self.encode.compile('adam', 'mse')
        self.decode.compile('adam', 'mse')

        if N_GPUS != 1:
            self.autoencoder = multi_gpu_model(self.autoencoder, N_GPUS)

        self.autoencoder.compile(optimizer, None)

        # add metrics
        for key in self.metrics:
            self.autoencoder.metrics_names.append(key)
            self.autoencoder.metrics_tensors.append(self.metrics[key])

    def wae_mmd(self, sample_qz):
        sample_pz = K.random_normal(shape=[self.p["batch_size"], self.p["latent_dim"]], mean=0.0,
                                    stddev=self.p["stddev"],
                                    dtype=tf.float32)

        s = 10
        # batch size
        n = int(self.p["batch_size"])

        ind = np.linspace(0, n - 1, n)
        ind = np.array([i + n * ind for i in ind]).flatten().astype(dtype=np.int)
        ind = np.expand_dims(ind, axis=1)

        def K_MAT(A, B):
            # A = K.reshape(A, [n, self.p["latent_dim"]])  # give tensorflow shape hints
            A = K.tile(A, [n, 1])
            B = K.tile(B, [n, 1])

            # indices to reshuffle A
            A = tf.gather_nd(A, indices=ind)
            # A = A[ind, :]
            # A should now be a matrix whose rows are [z1, ..., z1, z2, ..., z2...] etc
            distances = K.sqrt(K.sum(K.square(A - B), axis=1))
            distances = debugPrint(distances, "DISTANCES: ")
            # Returns a vector of n^2 distances
            # mat = tf.matmul(A - B, K.transpose(A - B))

            return K.exp((-0.5 / s ** 2) * distances)

        def K_MAT2(A, B):
            '''

            :param A: Matrix of size [n, D] - each row vector is sample
            :param B: Matrix of size [n, D] - each row vector is a sample
            :return: Matrix of size [n, n] - each entry (i,j) is k(a_i,b_j), where
            a_i is the i'th row of A and b_j is the j'th row of B
            '''
            # A = K.transpose(A)
            B = K.transpose(B)
            k_vecs = []
            for i in range(n):
                a_i = K.expand_dims(A[i, :])  # i'th row of first matrix
                Ai = K.tile(a_i, [1, n])  # n copies - size [D, n]

                # Will be a row vector of distances
                # Element j of dists_i = ||ai - bj||
                dists_i = K.sum(K.square(Ai - B), axis=0)
                dists_i = debugPrint(dists_i, "DIST {}".format(i))
                k_vec = K.exp((-1 / s ** 2) * dists_i)
                k_vecs.append(K.expand_dims(k_vec))

            k_mat = K.concatenate(k_vecs, axis=1)

            return K.transpose(k_mat)

        # k = lambda z1, z2: K.exp(-K.sum(K.square(z1 - z2), axis=1) / (s ** 2))

        ## First term
        ## 1/n(n+1) sum k(zp_l, zp_j) for all l=/=j

        ## Second term
        ## 1/n(n+1) sum k(zq_l, zq_j) for all l=/=j

        # Set diagonals to zero
        MMD = []
        for samples in [sample_qz, sample_pz]:
            K_sum = tf.reduce_sum(K_MAT(samples, samples)) - n

            K_sum = debugPrint(K_sum, "K_SUM: ")
            # there will be n diagonals in this matrix equal to 1
            # these shouldn't be included in the sum anyway
            # so simply subtract n after reduce_sum
            MMD.append(K_sum / (n * n + n))

        # Final term
        # -2/n * sum k(zp_l, zq_j) for all l, j
        MMD.append(-2 * tf.reduce_sum(K_MAT(sample_pz, sample_qz)) / n ** 2)

        return self.kl_loss_var * tf.reduce_sum(MMD)

    def mmd_penalty(self, sample_qz):
        '''
        :param stddev:
        :param kernel: RBF or IMQ
        :param pz: for IMQ kernel: 'normal', 'sphere' or 'uniform'
        :param sample_qz:
        :param sample_pz:
        :return:
        '''
        kernel = self.p["WAE_kernel"]

        sample_pz = K.random_normal(shape=[self.p["batch_size"], self.p["latent_dim"]], mean=0.0,
                                    stddev=self.p["stddev"],
                                    dtype=tf.float32)
        sigma2_p = self.p["WAE_s"] ** 2
        n = self.p["batch_size"]
        # n = tf.cast(n, tf.int32)
        nf = float(n)  # tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + K.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + K.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        if kernel == 'RBF':
            print("Using RBF WAE loss (s = {})".format(self.p["WAE_s"]))
            # Median heuristic for the sigma^2 of Gaussian kernel
            hs = int(half_size)  # tf.cast(half_size, tf.int32)
            sigma2_k = tf.nn.top_k(K.flatten(distances), hs).values[hs - 1]
            sigma2_k += tf.nn.top_k(K.flatten(distances_qz), hs).values[hs - 1]

            # Maximal heuristic for the sigma^2 of Gaussian kernel
            distances_qz /= sigma2_k
            distances_pz /= sigma2_k
            distances /= sigma2_k
            res1 = K.exp(- 0.5 * distances_qz)
            res1 += K.exp(- 0.5 * distances_pz)

            res1 = tf.multiply(res1, 1. - K.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = K.exp(-0.5 * distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif "IMQ" in kernel:
            pz = kernel.split("_")[1]
            print("Using IMQ loss with {} Cbase(s = {})".format(pz, self.p["WAE_s"]))
            if pz == 'normal':
                Cbase = 2. * self.p["latent_dim"] * sigma2_p
            elif pz == 'sphere':
                Cbase = 2.
            elif pz == 'uniform':
                Cbase = self.p["latent_dim"]
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2

        return self.kl_loss_var * stat

    def wae_mmd_exact(self, args):
        mu, logvar = args
        var = K.exp(logvar)
        s = self.p["WAE_s"]
        s2 = s ** 2
        s4 = s ** 4
        prior_mu = K.zeros_like(mu)
        prior_var = K.ones_like(var)

        def expected_rbf(mx, vx, my=None, vy=None):
            if my == None: my = mx
            if vy == None: vy = vx

            vxt = 1 / (1 / s2 + 1 / vx)
            vyt = 1 / (1 / s2 + 1 / vy + vxt / s4)
            myt = (my / vy - (mx * vxt) / (vx * s2)) * vyt

            det = lambda x: K.prod(x, axis=1)
            coeff = K.sqrt((det(vyt) * det(vxt)) / (det(vy) * det(vx)))

            exponent = K.square(mx) * vxt / K.square(vx)
            exponent += K.square(myt) / vyt
            exponent -= K.square(my) / vy
            exponent -= K.square(mx) / vx
            return coeff * K.exp(0.5 * K.sum(exponent, axis=1))

        return self.kl_loss_var * K.mean(expected_rbf(mu, var) +
                                         expected_rbf(prior_mu, prior_var) -
                                         2 * expected_rbf(mu, var, prior_mu, prior_var))


class TransformerEncoder():
    def __init__(self, params, tokens, pos_emb, word_emb):
        self.p = params
        self.encoder = tr.Encoder(params["d_model"], params["d_inner_hid"], params["heads"], params["d_k"],
                                  params["d_v"],
                                  params["layers"], params["dropout"], word_emb=word_emb, pos_emb=pos_emb)

        # # self.false_embedder = tr.FalseEmbeddings(d_emb=self.d_model)
        # if kl_loss_var is not None:
        #     stddev = params["stddev"] * kl_loss_var / params["kl_max_weight"]
        self.word_emb = word_emb
        if "average" in params["bottleneck"] or "sum" in params["bottleneck"]:
            self.encoder_to_latent = tr.latent_dict[params["bottleneck"]](params["d_model"], params["latent_dim"])
        elif "ar" in params["bottleneck"]:
            latent_pos_emb = tr.Embedding(params["latent_dim"], params["ID_d_model"], trainable=False)
            self.encoder_to_latent = tr.latent_dict[params["bottleneck"]](params["ID_d_model"],
                                                                          params["ID_d_inner_hid"],
                                                                          params["ID_heads"], params["ID_d_k"],
                                                                          params["ID_d_v"],
                                                                          params["ID_layers"], params["ID_width"],
                                                                          params["dropout"],
                                                                          stddev=1,
                                                                          latent_dim=params["latent_dim"],
                                                                          pos_emb=latent_pos_emb,
                                                                          false_emb=None)
        elif params["bottleneck"] == "trans_gru":
            self.encoder_to_latent = GRU(params["latent_dim"], return_sequences=False)
        elif params["bottleneck"] == "none":
            self.encoder_to_latent = tr.Vec2Variational(params["d_model"], params["len_limit"])

    def __call__(self, src_seq, src_pos):
        enc_output, enc_attn = self.encoder(src_seq,
                                            src_pos, return_att=True)

        # variational bottleneck produces [bs x latent_dim]
        z_s = None
        if self.p["bottleneck"] == "ar1":
            z_mean, z_logvar, z_s = self.encoder_to_latent(src_seq, enc_output)
        elif "ar" in self.p["bottleneck"]:
            z_mean, z_logvar = self.encoder_to_latent(src_seq, enc_output)
        elif self.p["bottleneck"] == "trans_gru":
            h = self.encoder_to_latent(enc_output)
            z_mean = Dense(self.p["latent_dim"])(h)
            z_logvar = Dense(self.p["latent_dim"])(h)
        else:
            z_mean, z_logvar = self.encoder_to_latent(enc_output)

        return z_mean, z_logvar, z_s, enc_attn


class SequenceInference():
    def __init__(self, model, tokens, weights_file=None):
        self.model = model
        self.tokens = tokens
        if weights_file:
            self.model.autoencoder.load_weights(weights_file, by_name=True)
            self.model.encode.load_weights(weights_file, by_name=True)
            self.model.decode.load_weights(weights_file, by_name=True)

        if model.p["model_arch"] == "TRANSFORMER" and model.p["bottleneck"] != "conv":
            self.prepare_str = lambda x: self.tokens.tokenize(x)
        else:
            self.prepare_str = lambda x: self.tokens.onehotify(x, model.p["len_limit"])

    def decode_sequence(self, input_seq, delimiter='', moments=None):
        # First get the latent representation
        target_seq = np.zeros((1, self.model.p["len_limit"]), dtype='int32')
        target_seq[0, 0] = self.tokens.startid()

        decoded_tokens = []
        # If mean/variance not provided, calculate
        if moments is None:
            src_seq = self.prepare_str(input_seq)
            [mean, logvar] = self.model.encode.predict_on_batch([src_seq, target_seq])
        else:
            mean, logvar = moments

        # sample from moments
        z = mean + np.exp(logvar) * np.random.normal(0, 1, np.shape(mean))

        for i in range(self.model.p["len_limit"] - 1):
            output = self.model.decode.predict_on_batch([z, target_seq])
            sampled_index = np.argmax(output[0, i, :])
            sampled_token = self.tokens.token(sampled_index)
            decoded_tokens.append(sampled_token)
            if sampled_index == self.tokens.endid(): break
            target_seq[0, i + 1] = sampled_index
        return delimiter.join(decoded_tokens[:-1])

    def decode_sequence_fast(self, input_seq, delimiter='', moments=None):
        '''
        Greedy decodes a sequence by keeping the most probable output symbol at each stage
        :param input_seq: String e.g. 'Cc1cccc1'
        :param delimiter:
        :return: output sequence as a string
        '''
        decoded_tokens = []
        target_seq = np.zeros((1, self.model.p["len_limit"]), dtype='int32')
        target_seq[0, 0] = self.tokens.startid()

        # If mean/variance not provided, calculate
        if moments is None:
            src_seq = self.prepare_str(input_seq)
            [mean, logvar] = self.model.encode.predict_on_batch([src_seq, target_seq])
        else:
            mean, logvar = moments

        # sample from moments
        z = mean + np.exp(logvar) * np.random.normal(0, 1, np.shape(mean))
        for i in range(self.model.p["len_limit"] - 1):
            output = self.model.decode.predict_on_batch([z, target_seq])
            sampled_index = np.argmax(output[0, i, :])
            sampled_token = self.tokens.token(sampled_index)
            decoded_tokens.append(sampled_token)
            if sampled_index == self.tokens.endid(): break
            target_seq[0, i + 1] = sampled_index
        return delimiter.join(decoded_tokens[:-1])

    def beam_search(self, input_seq=None, topk=5, delimiter='', moments=None):
        # If mean/variance not provided, calculate
        if moments is None:
            src_seq = self.prepare_str(input_seq)
            [mean, logvar] = self.model.encode.predict_on_batch(src_seq)
            z = mean + np.exp(logvar) * np.random.normal(0, 1, np.shape(mean))
        elif len(moments) == 2:
            # have been provided mean and variance
            mean, logvar = moments
            z = mean + np.exp(logvar) * np.random.normal(0, 1, np.shape(mean))
        else:
            # have been provided z
            z = np.reshape(moments[0], [1, len(moments[0])])

        z = z.repeat(topk, 0)

        final_results = []
        decoded_tokens = [[] for _ in range(topk)]
        decoded_logps = [0] * topk
        lastk = 1
        target_seq = np.zeros((topk, self.model.p["len_limit"]), dtype='int32')
        target_seq[:, 0] = self.tokens.startid()

        for i in range(self.model.p["len_limit"] - 1):
            if lastk == 0 or len(final_results) > topk * 3: break
            if self.model.p["model_arch"] == "TRANSFORMER":
                output = self.model.decode.predict_on_batch([z, target_seq])
            else:
                output = self.model.decode.predict_on_batch(z)
            output = np.exp(output[:, i, :])
            output = np.log(output / np.sum(output, -1, keepdims=True) + 1e-8)
            cands = []
            for k, wprobs in zip(range(lastk), output):
                if target_seq[k, i] == self.tokens.endid(): continue
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
                decoded_tokens.append(decoded_tokens[k] + [self.tokens.token(wid)])
                if wid == self.tokens.endid(): final_results.append((decoded_tokens[k], wprob))
            decoded_tokens = decoded_tokens[topk:]
            lastk = len(cands)
        final_results = [(x, y / (len(x) + 1)) for x, y in final_results]
        final_results.sort(key=lambda x: x[-1], reverse=True)
        final_results = [(delimiter.join(x), y) for x, y in final_results]
        return final_results
