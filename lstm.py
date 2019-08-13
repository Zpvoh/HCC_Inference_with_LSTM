import tensorflow as tf
import numpy as np


class LSTM:
    def __init__(self):
        self.num_units = 300
        self.vector_length = 166912
        self.cell_for = tf.contrib.rnn.BasicLSTMCell(self.num_units, state_is_tuple=True)
        self.cell_back = tf.contrib.rnn.BasicLSTMCell(self.num_units, state_is_tuple=True)
        self.input_tensor = tf.placeholder(tf.float32, [None, self.num_units, self.vector_length])
        tags_tensor = tf.placeholder(tf.int32, [None, self.num_units])
        self.seq_lengths_tensor = tf.placeholder(tf.int32, [None])

        out, states = tf.nn.bidirectional_dynamic_rnn(self.cell_for, self.cell_back, self.input_tensor,
                                                      self.seq_lengths_tensor,
                                                      dtype=tf.float32)
        output_fw, output_bw = out
        self.output = tf.concat([output_fw, output_bw], axis=-1)
