import tensorflow as tf
import functools


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class ModelLSTM(object):

    def __init__(self, data, labels, conf):
        self.data = data
        self.labels = labels
        self.conf = conf

    @lazy_property
    def data(self):
        return self.data

    @lazy_property
    def labels(self):
        return self.labels

    @lazy_property
    def conf(self):
        return self.conf

    @lazy_property
    def inference(self):
        data = tf.placeholder(tf.float32, [None, self.conf['sequence_length'], self.conf['number_of_features']])
        labels = tf.placeholder(tf.float32, [None, self.conf['number_of_features']])
        cell = tf.contrib.rnn.BasicLSTMCell(self.conf['number_hidden'], state_is_tuple = True)
        val, _ = tf.nn.dynamic_rnn(cell, data, dtype = tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        last_activated = tf.nn.relu(last)

        weight = tf.Variable(tf.truncated_normal([self.conf['number_hidden'], int(labels.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape = [labels.get_shape()[1]]))
        return tf.matmul(last_activated, weight) + bias

    @lazy_property
    def optimzer(self):
        l2_loss = tf.reduce_mean(tf.squared_difference(self.inference, self.labels))
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(l2_loss)

    @lazy_property
    def error(self):
        pass

