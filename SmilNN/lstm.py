# -*- coding: utf-8 -*-

from keras.layers.recurrent import *


class LSTM_base(Recurrent):
    '''Long Short-Term Memory unit - Hochreiter 1997.
    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Clockwork RNN](http://arxiv.org/abs/1402.3511)
    '''

    def __init__(self, output_dim,
                 num_blocks=None, connection='full',
                 with_recurrent_link=True, with_i=True, with_f=True, with_o=True,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.num_blocks = output_dim if num_blocks is None else num_blocks
        self.connection = connection
        self.with_recurrent_link = with_recurrent_link
        self.with_i, self.with_f, self.with_o = with_i, with_f, with_o
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(LSTM_base, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.connection in ['full', 'clockwork']
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        self.rep = self.output_dim / self.num_blocks
        if self.with_recurrent_link:
            if self.connection == 'clockwork':
                self.U_mask = K.variable(np.eye(self.num_blocks, k=1, dtype=K.floatx()))
                self.U_g_mask = K.variable(
                    np.repeat(np.repeat(np.eye(self.num_blocks, k=1, dtype=K.floatx()), self.rep, axis=0),
                               self.rep, axis=1))
            elif self.connection == 'full':
                self.U_mask = K.ones((self.num_blocks, self.num_blocks))
                self.U_g_mask = K.ones((self.output_dim, self.output_dim))

        self.trainable_weights = []
        self.W_g = self.init((input_dim, self.output_dim), name='{}_W_g'.format(self.name))
        self.trainable_weights.append(self.W_g)
        if self.with_recurrent_link:
            self.U_g = self.inner_init((self.output_dim, self.output_dim), name='{}_U_g'.format(self.name))
            self.trainable_weights.append(self.U_g)
        self.b_g = K.zeros((self.output_dim,), name='{}_b_g'.format(self.name))
        self.trainable_weights.append(self.b_g)

        if self.with_i:
            self.W_i = self.init((input_dim, self.num_blocks), name='{}_W_i'.format(self.name))
            self.trainable_weights.append(self.W_i)
            if self.with_recurrent_link:
                self.U_i = self.inner_init((self.num_blocks, self.num_blocks), name='{}_U_i'.format(self.name))
                self.trainable_weights.append(self.U_i)
            self.b_i = K.zeros((self.num_blocks,), name='{}_b_i'.format(self.name))
            self.trainable_weights.append(self.b_i)
        if self.with_f:
            self.W_f = self.init((input_dim, self.num_blocks), name='{}_W_f'.format(self.name))
            self.trainable_weights.append(self.W_f)
            if self.with_recurrent_link:
                self.U_f = self.inner_init((self.num_blocks, self.num_blocks), name='{}_U_f'.format(self.name))
                self.trainable_weights.append(self.U_f)
            self.b_f = self.forget_bias_init((self.num_blocks,), name='{}_b_f'.format(self.name))
            self.trainable_weights.append(self.b_f)

        if self.with_o:
            self.W_o = self.init((input_dim, self.num_blocks), name='{}_W_o'.format(self.name))
            self.trainable_weights.append(self.W_o)
            if self.with_recurrent_link:
                self.U_o = self.inner_init((self.num_blocks, self.num_blocks), name='{}_U_o'.format(self.name))
                self.trainable_weights.append(self.U_o)
            self.b_o = K.zeros((self.num_blocks,), name='{}_b_o'.format(self.name))
            self.trainable_weights.append(self.b_o)

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_g,
                                                        self.W_i,
                                                        self.W_f,
                                                        self.W_o]))
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_g,
                                                        self.U_i,
                                                        self.U_f,
                                                        self.U_o]))
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_g,
                                                        self.b_i,
                                                        self.b_f,
                                                        self.b_o]))
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights


    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]


    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_c = time_distributed_dense(x, self.W_g, self.b_g, dropout,
                                         input_dim, self.output_dim, timesteps)
            result = K.concatenate([x_c], axis=2)
            if self.with_i:
                x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                             input_dim, self.output_dim, timesteps)
                result = K.concatenate([result, x_i], axis=2)
            if self.with_f:
                x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                             input_dim, self.output_dim, timesteps)
                result = K.concatenate([result, x_f], axis=2)
            if self.with_o:
                x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                             input_dim, self.output_dim, timesteps)
                result = K.concatenate([result, x_o], axis=2)
            return result
        else:
            return x


    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_W = states[2]

        if self.with_recurrent_link:
            B_U = states[3]
            if self.with_i:
                U_i = K.repeat_elements(self.U_i * self.U_mask, self.rep, axis=0)
            if self.with_f:
                U_f = K.repeat_elements(self.U_f * self.U_mask, self.rep, axis=0)
            if self.with_o:
                U_o =K.repeat_elements(self.U_o * self.U_mask, self.rep, axis=0)
        if self.with_i:
            W_i = K.repeat_elements(self.W_i, self.rep, axis=0)
            b_i = K.repeat_elements(self.b_i, self.rep, axis=0)
        if self.with_f:
            W_f = K.repeat_elements(self.W_f, self.rep, axis=0)
            b_f = K.repeat_elements(self.b_f, self.rep, axis=0)
        if self.with_o:
            W_o = K.repeat_elements(self.W_o, self.rep, axis=0)
            b_o = K.repeat_elements(self.b_o, self.rep, axis=0)

        if self.consume_less == 'cpu':
            x_g = x[:, :self.output_dim]
            last = self.output_dim
            if self.with_i:
                x_i = K.repeat_elements(x[:, last: last + self.num_blocks], self.rep, axis=0)
                last = last + self.num_blocks
            if self.with_f:
                x_f = K.repeat_elements(x[:, last: last + self.output_dim], self.rep, axis=0)
                last = last + self.num_blocks
            if self.with_o:
                x_o = K.repeat_elements(x[:, last: last + self.output_dim], self.rep, axis=0)
        else:
            x_g = K.dot(x * B_W[0], self.W_g) + self.b_g
            last = 0
            if self.with_i:
                x_i = K.dot(x * B_W[last + 1], W_i) + b_i
                last += 1
            if self.with_f:
                x_f = K.dot(x * B_W[last + 1], W_f) + b_f
                last += 1
            if self.with_o:
                x_o = K.dot(x * B_W[last + 1], W_o) + b_o
                last += 1

        if self.with_recurrent_link:
            u_g = self.activation(K.dot(h_tm1 * B_U[0], self.U_g * self.U_g_mask))
            last = 0
            if self.with_i:
                u_i = K.dot(h_tm1 * B_U[last + 1], U_i)
                last += 1
            if self.with_f:
                u_f = K.dot(h_tm1 * B_U[last + 1], U_f)
                last += 1
            if self.with_o:
                u_o = K.dot(h_tm1 * B_U[last + 1], U_o)
                last += 1
            g = self.activation(x_g + u_g)
            gi = self.inner_activation(x_i + u_i) * g if self.with_i else g
            c = self.inner_activation(x_f + u_f) * c_tm1 + gi if self.with_f else c_tm1 + gi
            h = self.inner_activation(x_o + u_o) * self.activation(c) if self.with_o else self.activation(c)
        else:
            g = self.activation(x_g)
            gi = self.inner_activation(x_i) * g if self.with_i else g
            c = self.inner_activation(x_f) * c_tm1 + gi if self.with_f else c_tm1 + gi
            h = self.inner_activation(x_o) * self.activation(c) if self.with_o else self.activation(c)
        return h, [h, c]


    def get_constants(self, x):
        constants = []

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_U < 1 and self.with_recurrent_link:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        return constants


    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "num_blocks": self.num_blocks,
                  "with_recurrent_link": self.with_recurrent_link,
                  "connection": self.connection,
                  "with_i": self.with_i,
                  "with_f": self.with_f,
                  "with_o": self.with_o,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(LSTM_base, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
