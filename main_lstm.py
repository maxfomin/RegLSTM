import data_generator
import database
import tensorflow as tf
import yaml


def main():
    with open('definitions.yml', 'r') as f:
        conf = yaml.load(f)

    data, labels = data_generator.generate_data()
    data_struct = database.DataStruct(data, labels)

    data = tf.placeholder(tf.float32, [None, conf['sequence_length'], conf['number_of_features']])
    labels = tf.placeholder(tf.float32, [None, conf['number_of_features']])
    cell = tf.contrib.rnn.BasicLSTMCell(conf['number_hidden'], state_is_tuple = True)
    val, _ = tf.nn.dynamic_rnn(cell, data, dtype = tf.float32)

    weight = tf.Variable(tf.truncated_normal([conf['number_hidden'], int(labels.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape = [labels.get_shape()[1]]))
    pass

if __name__ == '__main__':
    main()
