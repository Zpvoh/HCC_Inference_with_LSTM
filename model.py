import tensorflow as tf
from lstm import LSTM


class Model:
    def __init__(self, learning_rate, length, tags_num):
        # self.ser_a = tf.placeholder(tf.float32, [None, 20], name="series_a")
        # self.ser_v = tf.placeholder(tf.float32, [None, 20], name="series_v")
        # self.ser_d = tf.placeholder(tf.float32, [None, 20], name="series_d")
        self.tags_num = tags_num
        self.length = length
        self.feed_dict = {}
        self.lstm = LSTM()
        self.y_true = tf.placeholder(tf.int32, [self.length, self.tags_num], name="y_true")
        output = self.lstm.output
        # self.W = tf.Variable(tf.random_normal(shape=[self.lstm.num_units*2, 2], mean=0.0, stddev=1.0))
        # self.b = tf.Variable(tf.random_normal(shape=[self.length, 2], mean=0.0, stddev=1.0))
        # linear_out = tf.matmul(output, self.W) + self.b
        W = tf.get_variable("W", [2 * self.lstm.num_units, self.tags_num], dtype=tf.float32)
        matricized_output = tf.reshape(output, [-1, 2 * self.lstm.num_units])
        matricized_unary_scores = tf.matmul(matricized_output, W)
        unary_scores = tf.reshape(matricized_unary_scores, [-1, self.lstm.vector_length, self.tags_num])

        self.soft = tf.nn.softmax(matricized_unary_scores)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=matricized_unary_scores))
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

    def init_seq(self, input_lstm, seq_len):
        self.feed_dict[self.lstm.input_tensor] = input_lstm
        self.feed_dict[self.lstm.seq_lengths_tensor] = seq_len

    # def init_seq_v(self, input_v, seq_len_v):
    #     self.feed_dict[self.lstm_v.input_tensor] = input_v
    #     self.feed_dict[self.lstm_v.seq_lengths_tensor] = seq_len_v
    #
    # def init_seq_d(self, input_d, seq_len_d):
    #     self.feed_dict[self.lstm_d.input_tensor] = input_d
    #     self.feed_dict[self.lstm_d.seq_lengths_tensor] = seq_len_d

    def init_y_true(self, y_true):
        self.feed_dict[self.y_true] = y_true

    def train(self, batch_size, input_a, seq_len_a, y_true):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.init_seq(input_a, seq_len_a)
        self.init_y_true(y_true)
        _, loss = sess.run([self.loss], feed_dict=self.feed_dict)
        print("loss is " + str(loss))
