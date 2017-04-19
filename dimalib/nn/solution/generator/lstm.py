import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import rnn
import sys


class LSTMCharGenerator:
    def __init__(self, vocab=None, lstm_size=256, batch_len=30, batch_size=128,
                 learning_rate=0.01, regularization=True, reg_coef=0.0001, verbose=True):
        self.text = ""
        if vocab is not None:
            self.vocab = list(set(vocab))
        else:
            self.vocab = None
        self.lstm_size = lstm_size
        self.max_len = batch_len
        self.batch_size = batch_size
        self.segment_size = 0
        self.segment_pos = 0
        self.regularization = regularization
        self.reg_coef = reg_coef
        self.graph = None
        self.verbose = verbose
        self.loss = 0
        self.last_sample = ""
        if self.vocab is not None:
            self.vocab_len = len(self.vocab)
            self.vocab_dict = dict([(x, i) for (i, x) in enumerate(self.vocab)])
            self.rev_vocab = dict([(i, x) for (x, i) in self.vocab_dict.items()])
        self.vocab_len = 0
        self.cur_iter = 0
        self.max_iter = 0
        self.learning_rate = learning_rate
        self.prev_sample = ""

    def fit(self, text, max_iter=10):
        self.text = text
        self.segment_pos = 0
        self.loss = 0
        self.cur_iter = 0
        self.max_iter = max_iter
        self.segment_size = len(text) // self.batch_size
        if self.vocab is None:
            self.vocab = list(set(text))
            self.vocab_dict = dict([(x, i) for (i, x) in enumerate(self.vocab)])
            self.rev_vocab = dict([(i, x) for (x, i) in self.vocab_dict.items()])
            self.vocab_len = len(self.vocab)
        if self.graph is None:
            self.build_graph()
            self.sess = tf.InteractiveSession(graph=self.graph)
            tf.global_variables_initializer().run()
        for i in range(max_iter):
            self.iter_print_repeat()

    def iter_print_repeat(self):
        cur_loss = self.single_iter()
        self.cur_iter += 1
        if self.verbose:
            if self.loss == 0:
                self.loss = cur_loss
            self.loss = self.loss * 0.99 + cur_loss * 0.01
            seed_start = random.randrange(len(self.text) - 4)
            seed = self.text[seed_start:seed_start + 2]
            if self.cur_iter % 50 == 0:
                self.prev_sample = self.generate_sample_with_seed(40, seed, 0.1).replace("\r", "").replace("\n", "")
            msg = "Loss:" + str(self.loss) + ". Sample pred: " + str(self.prev_sample)
            progress = progress_bar(self.cur_iter, self.max_iter, 30, msg)
            verbose_len = 140
            progress = progress[0:verbose_len]
            for i in range(verbose_len - len(progress)):
                progress += " "
            if self.cur_iter % 3 == 0:
                sys.stdout.write('\r' + progress)

    def single_iter(self):
        x_batch = self.next_batch()
        _, ll = self.sess.run([self.optimizer_func, self.loss_func], feed_dict={
            self.x_input: x_batch,
        })
        return ll

    def build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            w = tf.Variable(tf.truncated_normal([self.lstm_size, self.vocab_len], -0.1, 0.1))
            b = tf.Variable(tf.zeros([self.vocab_len]))
            lstm_cell = rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0)

            saved_state = (tf.Variable(tf.zeros([self.batch_size, self.lstm_size]), trainable=False),
                           tf.Variable(tf.zeros([self.batch_size, self.lstm_size]), trainable=False))
            x_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_len, self.vocab_len])
            x = tf.unstack(x_input, self.max_len, 1)
            outputs = []
            state = saved_state
            with tf.variable_scope("rnn") as scope:
                for i in x[:-1]:
                    output, state = lstm_cell(i, state)
                    scope.reuse_variables()
                    outputs.append(output)
            labels = x[1:]
            with tf.control_dependencies([saved_state[0].assign(state[0]),
                                          saved_state[1].assign(state[1])]):
                logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat(labels, 0), logits=logits))
                preds = tf.nn.softmax(logits)
                regularization = tf.constant(0.0, dtype=tf.float32)
                reg_param = tf.constant(self.reg_coef, dtype=tf.float32)
                if self.regularization:
                    for v in tf.trainable_variables():
                        regularization += tf.nn.l2_loss(v) * reg_param
                loss = loss + regularization

            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            sample_input = tf.placeholder(tf.float32, shape=[1, self.vocab_len])
            saved_sample_state = (tf.Variable(tf.zeros([1, self.lstm_size]), trainable=False),
                                  tf.Variable(tf.zeros([1, self.lstm_size]), trainable=False))
            reset_sample_state = tf.group(
                saved_sample_state[0].assign(tf.zeros([1, self.lstm_size])),
                saved_sample_state[1].assign(tf.zeros([1, self.lstm_size]))
            )

            with tf.variable_scope("rnn") as scope:
                scope.reuse_variables()
                sample_output, sample_state = lstm_cell(sample_input, saved_sample_state)
            with tf.control_dependencies([saved_sample_state[0].assign(sample_state[0]),
                                          saved_sample_state[1].assign(sample_state[1])]):
                sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
        self.graph = graph
        self.optimizer_func = optimizer
        self.loss_func = loss
        self.reset_sample_state_func = reset_sample_state
        self.x_input = x_input
        self.sample_prediction_func = sample_prediction
        self.sample_input = sample_input

    def generate_sample_with_seed(self, sample_len, seed, diversity=0.1):
        result_str = seed
        self.sess.run(self.reset_sample_state_func)
        for i, letter in enumerate(seed[:-1]):
            mat = np.zeros([1, self.vocab_len])
            mat[0, self.char2id(letter)] = 1.0
            pred = self.sess.run(self.sample_prediction_func, feed_dict={
                self.sample_input: mat,
            })
        prev_letter = seed[-1]
        for i in range(sample_len):
            mat = np.zeros([1, self.vocab_len])
            mat[0, self.char2id(prev_letter)] = 1.0
            pred = self.sess.run(self.sample_prediction_func, feed_dict={
                self.sample_input: mat,
            })
            let = self.sample(pred[0], diversity)
            prev_letter = self.id2char(let)
            result_str += prev_letter
        return result_str

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def next_batch(self):
        batch = np.zeros([self.batch_size, self.max_len, self.vocab_len])
        for i in range(self.batch_size):
            for j in range(self.max_len):
                idx = (i * self.segment_size + self.segment_pos + j) % len(self.text)
                cur_chr = self.vocab_dict.get(self.text[idx], None)
                if cur_chr is not None:
                    batch[i, j, cur_chr] = 1.0
        self.segment_pos += self.max_len - 1
        return batch

    def char2id(self, char):
        return self.vocab_dict[char]

    def id2char(self, dictid):
        return self.rev_vocab[dictid]


def progress_bar(cur_iter, max_iter, bar_len, msg):
    fraction = cur_iter / max_iter
    progress_end = int(bar_len * fraction)
    result_str = "["
    for i in range(progress_end - 1):
        result_str += "="
    result_str += ">"
    for i in range(progress_end, bar_len):
        result_str += "."
    result_str += "]"
    iters_str = str(cur_iter) + "/" + str(max_iter)
    result_str += "  " + iters_str + "  "
    return result_str + msg
