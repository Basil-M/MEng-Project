import copy
import tensorflow as tf
import numpy as np

from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Softmax
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
import molecules.transformer as tr
from molecules.transformer import debugPrint
from keras.utils.training_utils import multi_gpu_model


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
class TriTransformer:
    def __init__(self, i_tokens, p, o_tokens=None):
        self.i_tokens = i_tokens
        # self.params = p
        self.p = lambda param: p.get(param)

        if o_tokens is None:
            # Autoencoder
            o_tokens = i_tokens

        self.o_tokens = o_tokens
        self.src_loc_info = True
        self.decode = None

        self.pp_layers = self.p("pp_layers")
        d_emb = self.p("d_model")

        pos_emb = tr.Embedding(self.p("len_limit"), d_emb, trainable=False,
                               weights=[tr.GetPosEncodingMatrix(self.p("len_limit"), d_emb)])
        i_word_emb = tr.Embedding(i_tokens.num(), d_emb)
        o_word_emb = i_word_emb

        self.encoder = tr.Encoder(self.p("d_model"), self.p("d_inner_hid"), self.p("heads"), self.p("d_k"),
                                  self.p("d_v"),
                                  self.p("layers"), self.p("dropout"), word_emb=i_word_emb, pos_emb=pos_emb)

        # self.false_embedder = tr.FalseEmbeddings(d_emb=self.d_model)
        if self.p("stddev") == 0 or self.p("stddev") is None:
            self.kl_loss_var = None
        else:
            self.kl_loss_var = K.variable(0.0, dtype=np.float, name='kl_loss_weight')
            self.stddev = self.p("stddev") * self.kl_loss_var / self.p("kl_max_weight")

        self.false_embedder = tr.FalseEmbeddingsNonTD(d_emb=self.p("d_model"), d_latent=self.p("latent_dim"))

        if self.p("bottleneck") == "average":
            self.encoder_to_latent = tr.AvgLatent(self.p("d_model"), self.p("latent_dim"))
        elif self.p("bottleneck") == "interim_decoder":
            latent_pos_emb = tr.Embedding(self.p("latent_dim"), self.p("ID_d_model"), trainable=False)
            self.encoder_to_latent = tr.InterimDecoder3(self.p("ID_d_model"), self.p("ID_d_inner_hid"),
                                                        self.p("ID_heads"), self.p("ID_d_k"), self.p("ID_d_v"),
                                                        self.p("ID_layers"), self.p("ID_width"), self.p("dropout"),
                                                        stddev=self.stddev,
                                                        latent_dim=self.p("latent_dim"),
                                                        pos_emb=latent_pos_emb,
                                                        false_emb=None)

        elif self.p("bottleneck") == "none":
            self.encoder_to_latent = tr.Vec2Variational(self.p("d_model"), self.p("len_limit"))

        # Sample from latent space
        def sampling(args):
            z_mean_, z_logvar_ = args
            if self.p("stddev") is None or self.p("stddev") == 0:
                return z_mean_
            else:
                if self.p("bottleneck") == "none":
                    latent_shape = (K.shape(z_mean_)[0], K.shape(z_mean_)[1], self.p("d_model"))
                else:
                    latent_shape = (K.shape(z_mean_)[0], self.p("latent_dim"))

                epsilon = K.random_normal(shape=latent_shape, mean=0., stddev=self.stddev)
                return z_mean_ + K.exp(z_logvar_ / 2) * epsilon

        self.sampler = Lambda(sampling)
        self.decoder = tr.DecoderFromLatent(self.p("d_model"), self.p("d_inner_hid"), self.p("heads"), self.p("d_k"),
                                            self.p("d_v"),
                                            self.p("layers"), self.p("dropout"),
                                            word_emb=o_word_emb, pos_emb=pos_emb)
        self.target_layer = TimeDistributed(Dense(o_tokens.num(), use_bias=False))
        # self.target_softmax = Softmax()

        if self.p("pp_weight") == 0 or self.p("pp_weight") is None:
            self.pp_loss_var = None
            print("Not joint training property encoder")
        else:
            self.pp_loss_var = K.variable(self.p("pp_weight"), dtype=np.float, name='pp_loss_weight')
            print("Joint training property encoder with PP weight {}".format(self.p("pp_weight")))

        self.metrics = {}

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def build_models(self, active_layers=999):
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

        if self.p("bottleneck") == "interim_decoder":
            z_mean, z_logvar = self.encoder_to_latent(src_seq, enc_output)
        else:
            z_mean, z_logvar = self.encoder_to_latent(enc_output)

        # sample from z_mean, z_logvar
        z_sampled = self.sampler([z_mean, z_logvar])
        if self.p("bottleneck") == "none":
            z_input = Input(shape=(None, self.p("d_model")), dtype='float32', name='z_input')
        else:
            z_input = Input(shape=(self.p("latent_dim"),), dtype='float32', name='z_input')

        # must calculate for both a latent input and the full end-to-end system
        final_output = []
        for l_vec in [z_input, z_sampled]:
            # 'false embed' for decoder
            dec_input = self.false_embedder(l_vec)

            dec_input = debugPrint(dec_input, "DEC_INPUT (Embedded sampled z)")

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
        if self.p("latent_dim") is None:
            latent_dim = self.p("d_model") * self.p("len_limit")
            latent_vec = Input(shape=(self.p("d_model"), self.p("len_limit"),), dtype='float', name='latent_rep')
        else:
            latent_dim = self.p("latent_dim")
            latent_vec = Input(shape=[latent_dim], dtype='float', name='latent_rep')

        if self.pp_loss_var is not None:
            # latent_vec = Lambda(lambda x: K.reshape(x, [-1, latent_dim]))(z_sampled)
            latent_vec = z_sampled

        h = Dense(latent_dim, input_shape=(latent_dim,), activation='linear')(latent_vec)
        num_props = 4
        prop_input = Input(shape=[num_props])
        for _ in range(self.p("pp_layers") - 1):
            h = Dense(latent_dim, activation='linear')(h)
        prop_output = Dense(num_props, activation='linear')(h)

        # RECONSTRUCTION LOSS
        def rec_loss(args):
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            return K.mean(loss)

        # KL DIVERGENCE LOSS
        def kl_loss(args):
            z_mean_, z_log_var_ = args

            return - 0.5 * self.kl_loss_var * tf.reduce_mean(1 + z_log_var_ - K.square(z_mean_) - K.exp(z_log_var_),
                                                             name='KL_loss_sum')

        # PROPERTY PREDICTION LOSS
        def pp_loss(args):
            prop_input_, prop_output_ = args
            SE = K.pow(prop_input_ - prop_output_, 2)
            SE = K.sqrt(SE)
            return self.pp_loss_var * tf.reduce_mean(SE)  # / K.shape(prop_output)[0]

        def get_accu(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        losses = []
        losses.append(Lambda(rec_loss, name='ReconstructionLoss')([final_output[1], tgt_true]))

        # WASSERSTEIN LOSS
        if self.kl_loss_var is not None:
            if self.p("RBF_s") == 0:
                print("Using variational autoencoder")
                kl = Lambda(kl_loss, name='VariationalLoss')([z_mean, z_logvar])
            else:
                print("Using Wasserstein autoencoder, with RBF kernel (s = {})".format(self.p("RBF_s")))
                kl = Lambda(self.wae_mmd, name='VariationalLoss')(z_sampled)
            self.metrics["kl_loss"] = kl
            losses.append(kl)

        # PROPERTY PREDICTION LOSS
        if self.pp_loss_var is not None:
            pp = Lambda(pp_loss, name='PropertyLoss')([prop_input, prop_output])
            losses.append(pp)
            self.metrics['pp_loss'] = pp

        loss = Lambda(tf.reduce_sum, name='LossesSum')(losses)

        # Set up autoencoder model
        if self.pp_loss_var is None:
            self.autoencoder = Model(src_seq_input, loss)
        else:
            self.autoencoder = Model([src_seq_input, prop_input], loss)

        self.autoencoder.add_loss([loss])

        ## METRICS
        self.metrics["ppl"] = Lambda(K.exp)(loss)
        self.metrics["accu"] = Lambda(get_accu)([final_output[1], tgt_true])
        self.metrics["meanmean"] = Lambda(tf.reduce_mean)(z_mean)
        self.metrics["meanlogvar"] = Lambda(tf.reduce_mean)(z_logvar)

        if tr.DEBUG:
            self.autoencoder.metrics_names.append('lr')
            self.autoencoder.metrics_tensors.append(self.autoencoder.optimizer.lr)

        # For outputting next symbol
        self.output_model = Model(src_seq_input, final_output[1])

        # For getting attentions
        attn_list = enc_attn + dec_attn + encdec_attn
        self.output_attns = Model(src_seq_input, attn_list)

        # ENCODING/DECODING MODELS
        self.encode = Model(src_seq_input, [z_mean, z_logvar])
        self.encode_sample = Model(src_seq_input, [z_sampled])
        self.decode = Model([z_input, tgt_seq_input], final_output[0])


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

    def make_src_seq_matrix(self, input_seq):
        '''
        Given a string input sequence will return a tokenised sequence
        :param input_seq:
        :return:
        '''
        if isinstance(input_seq, str):
            src_seq = np.zeros((1, len(input_seq) + 3), dtype='int32')
            src_seq[0, 0] = self.i_tokens.startid()
            for i, z in enumerate(input_seq):
                src_seq[0, 1 + i] = self.i_tokens.id(z)
            src_seq[0, len(input_seq) + 1] = self.i_tokens.endid()
        else:
            src_seq = np.expand_dims(input_seq,0)

        return src_seq

    def decode_sequence(self, input_seq, delimiter='', moments=None):
        # First get the latent representation
        target_seq = np.zeros((1, self.p("len_limit")), dtype='int32')
        target_seq[0, 0] = self.o_tokens.startid()

        decoded_tokens = []
        # If mean/variance not provided, calculate
        if moments is None:
            src_seq = self.make_src_seq_matrix(input_seq)
            z = self.encode_sample.predict_on_batch([src_seq, target_seq])
        else:
            mean, logvar = moments
            z = mean + np.exp(logvar)*np.random.normal(0,1,np.shape(mean))

        for i in range(self.p("len_limit") - 1):
            output = self.decode.predict_on_batch([z, target_seq])
            sampled_index = np.argmax(output[0, i, :])
            sampled_token = self.o_tokens.token(sampled_index)
            decoded_tokens.append(sampled_token)
            if sampled_index == self.o_tokens.endid(): break
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
        target_seq = np.zeros((1, self.p("len_limit")), dtype='int32')
        target_seq[0, 0] = self.o_tokens.startid()

        # If mean/variance not provided, calculate
        if moments is None:
            src_seq = self.make_src_seq_matrix(input_seq)
            z = self.encode_sample.predict_on_batch([src_seq, target_seq])
        else:
            mean, logvar = moments
            z = mean + np.exp(logvar)*np.random.normal(0,1,np.shape(mean))

        for i in range(self.p("len_limit") - 1):
            output = self.decode.predict_on_batch([z, target_seq])
            sampled_index = np.argmax(output[0, i, :])
            sampled_token = self.o_tokens.token(sampled_index)
            decoded_tokens.append(sampled_token)
            if sampled_index == self.o_tokens.endid(): break
            target_seq[0, i + 1] = sampled_index
        return delimiter.join(decoded_tokens[:-1])

    def beam_search(self, input_seq=None, topk=5, delimiter='', moments=None):
        # If mean/variance not provided, calculate
        if moments is None:
            src_seq = self.make_src_seq_matrix(input_seq)
            z = self.encode_sample.predict_on_batch(src_seq)
        else:
            mean, logvar = moments
            z = mean + np.exp(logvar) * np.random.normal(0, 1, np.shape(mean))

        z = z.repeat(topk, 0)

        final_results = []
        decoded_tokens = [[] for _ in range(topk)]
        decoded_logps = [0] * topk
        lastk = 1
        target_seq = np.zeros((topk, self.p("len_limit")), dtype='int32')
        target_seq[:, 0] = self.o_tokens.startid()

        for i in range(self.p("len_limit") - 1):
            if lastk == 0 or len(final_results) > topk * 3: break
            output = self.decode.predict_on_batch([z, target_seq])
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

    def wae_mmd(self, sample_qz):
        sample_pz = K.random_normal(shape=[self.p("batch_size"), self.p("latent_dim")], mean=0.0,
                                    stddev=self.p("stddev"),
                                    dtype=tf.float32)

        s = 10
        # batch size
        n = int(self.p("batch_size"))

        ind = np.linspace(0, n - 1, n)
        ind = np.array([i + n * ind for i in ind]).flatten().astype(dtype=np.int)
        ind = np.expand_dims(ind, axis=1)

        def K_MAT2(A, B):
            # A = K.reshape(A, [n, self.p("latent_dim")])  # give tensorflow shape hints
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

        def K_MAT(A, B):
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
        kernel = 'RBF'
        sample_pz = K.random_normal(shape=[self.p("batch_size"), self.p("latent_dim")], mean=0.0,
                                    stddev=self.p("stddev"),
                                    dtype=tf.float32)
        sigma2_p = self.p("RBF_s") ** 2
        n = self.p("batch_size")
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
            # Median heuristic for the sigma^2 of Gaussian kernel
            hs = int(half_size)  # tf.cast(half_size, tf.int32)
            # sigma2_k = K.flatten(distances) + K.flatten(distances_qz)
            sigma2_k = tf.nn.top_k(K.flatten(distances), hs).values[hs - 1]
            sigma2_k += tf.nn.top_k(K.flatten(distances_qz), hs).values[hs - 1]
            # Maximal heuristic for the sigma^2 of Gaussian kernel
            # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
            # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
            # sigma2_k = opts['latent_space_dim'] * sigma2_p
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
        elif kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            pz = 'normal'
            if pz == 'normal':
                Cbase = 2. * self.p("latent_dim") * sigma2_p
            elif pz == 'sphere':
                Cbase = 2.
            elif pz == 'uniform':
                Cbase = self.p("latent_dim")
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
