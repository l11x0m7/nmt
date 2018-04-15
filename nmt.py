# -*- encoding : utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import numpy as np
import cPickle as pkl

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def transform(data, word2id):
    ret_data = []
    for sentence in data:
        ret_data.append([word2id.get(word, 1) for word in sentence.split()])
    return ret_data

def padding(data, max_len):
    return tf.keras.preprocessing.sequence.pad_sequences(data, max_len, padding='post', truncating='post')


class Iterator(object):
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.sample_num = self.x.shape[0]

    def next_batch(self, batch_size):
        # produce X, Y_out, Y_in, X_len, Y_in_len, Y_out_len
        l = np.random.randint(0, self.sample_num - batch_size + 1)
        r = l + batch_size
        x_part = self.x[l:r]
        y_out_part = self.y[l:r]
        x_len = np.sum((x_part > 0), axis=1)
        y_in_part = np.concatenate((np.ones((batch_size, 1)) * 2, y_out_part[:,:-1]), axis=-1)
        max_y_dim = self.y.shape[1]
        y_out_len = np.sum((y_out_part > 0), axis=1) + 1
        y_out_len = np.min(np.concatenate([np.ones((batch_size, 1)) * max_y_dim, y_out_len.reshape(-1, 1)], axis=-1), axis=-1)
        y_in_len = np.sum((y_in_part > 0), axis=1) + 1
        y_in_len = np.min(np.concatenate([np.ones((batch_size, 1)) * max_y_dim, y_in_len.reshape(-1, 1)], axis=-1), axis=-1)
        x_len = x_len.astype(np.int32)
        y_in_part = y_in_part.astype(np.int32)
        y_in_len = y_in_len.astype(np.int32)
        y_out_len = y_out_len.astype(np.int32)
        return x_part, y_out_part, y_in_part, x_len, y_in_len, y_out_len


class NMTModel(object):
    def __init__(self, 
                 src_max_vocab_size, 
                 tgt_max_vocab_size, 
                 embedding_size,
                 hidden_size,
                 src_max_seq_len,
                 tgt_max_seq_len,
                 tgt_start_id,
                 tgt_end_id,
                 max_gradient_norm=5,
                 maximum_iterations=None
                 ):
        self.initializer = tf.random_uniform_initializer(
        -0.01, 0.01)
        self.src_max_vocab_size = src_max_vocab_size
        self.tgt_max_vocab_size = tgt_max_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.src_max_seq_len = src_max_seq_len
        self.tgt_max_seq_len = tgt_max_seq_len
        self.tgt_start_id = tgt_start_id
        self.tgt_end_id = tgt_end_id
        if maximum_iterations is None:
            self.maximum_iterations = self.tgt_max_seq_len
        else:
            self.maximum_iterations = maximum_iterations
        self.max_gradient_norm = max_gradient_norm
        self.add_placeholders()
        self.batch_size = tf.shape(self.X)[0]
        self.add_embeddings()
        self.encoder()
        self.decoder()
        self.add_loss()
        self.add_train_op()

    def add_placeholders(self):
        # X, Y_out, Y_in, X_len, Y_in_len, Y_out_len
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y_out = tf.placeholder(tf.int32, [None, None])
        self.Y_in = tf.placeholder(tf.int32, [None, None])
        self.X_len = tf.placeholder(tf.int32, [None, ])
        self.Y_in_len = tf.placeholder(tf.int32, [None, ])
        self.Y_out_len = tf.placeholder(tf.int32, [None, ])
        self.lr = tf.placeholder(tf.float32)
        self.dropout = tf.placeholder(tf.float32)

    def add_embeddings(self):
        with tf.variable_scope('embeddings', initializer=self.initializer):
            self.X_emb = tf.get_variable('X_emb', 
                shape=(self.src_max_vocab_size, self.embedding_size), 
                dtype=tf.float32)
            self.Y_emb = tf.get_variable('Y_emb', 
                shape=(self.tgt_max_vocab_size, self.embedding_size), 
                dtype=tf.float32)

            self.encoder_input = tf.nn.embedding_lookup(self.X_emb, self.X)
            self.decoder_input = tf.nn.embedding_lookup(self.Y_emb, self.Y_in)

    def encoder(self):
        with tf.variable_scope('encoder'):
            fw_encoder_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
            fw_encoder_cell = tf.contrib.rnn.DropoutWrapper(fw_encoder_cell, input_keep_prob=1-self.dropout)
            bw_encoder_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
            bw_encoder_cell = tf.contrib.rnn.DropoutWrapper(bw_encoder_cell, input_keep_prob=1-self.dropout)

            encoder_outputs, bi_last_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_encoder_cell, bw_encoder_cell, self.encoder_input, 
                    self.X_len, dtype=tf.float32)
            self.encoder_outputs = tf.concat(encoder_outputs, axis=-1)
            self.encoder_last_state = bi_last_state


    def decoder(self):
        with tf.variable_scope('decoder'):
            decoder_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=1-self.dropout)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                                    self.hidden_size, self.encoder_outputs,
                                    memory_sequence_length=self.X_len)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                                    decoder_cell, attention_mechanism,
                                    attention_layer_size=self.hidden_size)

            projection_layer = layers_core.Dense(
            self.tgt_max_vocab_size, use_bias=False)

        with tf.variable_scope('dynamic_decode'):
            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_input, tf.ones((self.batch_size, ), dtype=tf.int32) * self.tgt_max_seq_len, time_major=False)
            # Decoder
            decoder_initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32).clone(
                cell_state=self.encoder_last_state[0])
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, decoder_initial_state,
                output_layer=projection_layer)
            # Dynamic decoding
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            self.logits = outputs.rnn_output

        with tf.variable_scope('dynamic_decode', reuse=True):
            # Helper
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.Y_emb,
                start_tokens=tf.fill([self.batch_size], self.tgt_start_id),
                end_token=self.tgt_end_id)
            decoder_initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32).clone(
                cell_state=self.encoder_last_state[0])
            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, decoder_initial_state,
                output_layer=projection_layer)
            # Dynamic decoding
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=self.maximum_iterations)
            self.translations = outputs.sample_id

    def add_loss(self):
        target_weights = tf.sequence_mask(
                         self.Y_out_len, self.tgt_max_seq_len, dtype=self.logits.dtype)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                         labels=self.Y_out, logits=self.logits)
        self.loss_op = (tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size))

    def add_train_op(self):
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        # Optimization
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_op = optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)


def main():
    batch_size = 32
    num_epochs = 20
    iter_num = 20000
    lr = 1e-3
    dropout = 0.2

    src_max_vocab_size = 60000
    tgt_max_vocab_size = 5000
    embedding_size = 128
    hidden_size = 128
    src_max_seq_len = 40
    tgt_max_seq_len = 40
    tgt_start_id = 2 # <S>
    tgt_end_id = 0 # <PAD>
    max_gradient_norm = 5
    maximum_iterations = 40
    cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    cf.gpu_options.per_process_gpu_memory_fraction = 0.4

    # load data and dictionary
    with open('data/preprocess/demo_vocab_dict.pkl') as fr:
        en_word2id, en_id2word, ch_word2id, ch_id2word, \
        train_en_corpus, train_ch_corpus, test_en_corpus, test_ch_corpus = pkl.load(fr)

    train_en_corpus = padding(train_en_corpus, src_max_seq_len)
    train_ch_corpus = padding(train_ch_corpus, tgt_max_seq_len)

    data_iterator = Iterator(train_en_corpus, train_ch_corpus)

    with tf.Session(config=cf) as sess:
        model = NMTModel(src_max_vocab_size=src_max_vocab_size, 
                         tgt_max_vocab_size=tgt_max_vocab_size, 
                         embedding_size=embedding_size,
                         hidden_size=hidden_size,
                         src_max_seq_len=src_max_seq_len,
                         tgt_max_seq_len=tgt_max_seq_len,
                         tgt_start_id=tgt_start_id,
                         tgt_end_id=tgt_end_id,
                         max_gradient_norm=max_gradient_norm,
                         maximum_iterations=maximum_iterations)
        sess.run(tf.global_variables_initializer())
        for epoch in xrange(num_epochs):
            for iter_n in xrange(iter_num):
                X, Y_out, Y_in, X_len, Y_in_len, Y_out_len = data_iterator.next_batch(batch_size)
                # print(X.shape)
                # print(Y_out.shape)
                # print(Y_in.shape)
                # print(X_len.shape)
                # print(Y_in_len.shape)
                # print(Y_out_len.shape)
                loss, _ = sess.run([model.loss_op, model.train_op], 
                    feed_dict={ model.X:X,
                                model.Y_out:Y_out,
                                model.Y_in:Y_in, 
                                model.X_len:X_len,
                                model.Y_in_len:Y_in_len,
                                model.Y_out_len:Y_out_len,
                                model.lr:lr,
                                model.dropout:dropout})
                print('Loss:{}'.format(loss))


if __name__ == '__main__':
    main()











