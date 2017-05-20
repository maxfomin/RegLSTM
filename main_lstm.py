import data_generator
import database
import model
import tensorflow as tf
import yaml
import timeit


def main():
    with open('definitions.yml', 'r') as f:
        conf = yaml.load(f)

    data, labels = data_generator.generate_data(conf)
    data_struct = database.DataStruct(data, labels)
    del data, labels
    number_batches = int(conf['data_size'] * (1 - conf['test_percentage']) / conf['batch_size'])

    lstm_nn = model.ModelLSTM(conf)
    minimize = lstm_nn.optimizer
    error = lstm_nn.error
    initialize = lstm_nn.init

    sess = tf.Session()
    sess.run(initialize)

    for i in range(conf['number_epoch']):
        time_start = timeit.timeit()

        for j in range(number_batches):
            train_data, train_labels = data_struct.get_batch()
            sess.run(minimize, {data: train_data, labels: train_labels})

        test_data, test_labels = data_struct.get_test()
        epoch_error = sess.run(error, {data: test_data, labels: test_labels})
        time_stop = timeit.timeit()
        print('Epoch {0}: time is {1} sec, error is {2}\n'.format(i + 1, time_stop - time_start, epoch_error))


if __name__ == '__main__':
    main()
