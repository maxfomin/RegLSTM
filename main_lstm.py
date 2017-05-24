import data_generator
import database
import model
import tensorflow as tf
import yaml
import time


def main():
    with open('definitions.yml', 'r') as f:
        conf = yaml.load(f)

    data, labels = data_generator.generate_data()
    data_struct = database.DataStruct(data, labels)
    del data, labels
    print('Data has been generated\n')

    number_batches = int(conf['data_size'] * (1 - conf['test_percentage']) / conf['batch_size'])
    lstm_nn = model.ModelLSTM()
    minimize = lstm_nn.optimizer
    error = lstm_nn.error
    initialize = lstm_nn.init

    sess = tf.Session()
    sess.run(initialize)

    for i in range(conf['number_epoch']):
        time_start = time.clock()

        for j in range(number_batches):
            batch_data, batch_labels = data_struct.get_batch(j)
            sess.run(minimize, {lstm_nn.data: batch_data, lstm_nn.labels: batch_labels})

        test_data, test_labels = data_struct.get_test()
        epoch_error = sess.run(error, {lstm_nn.data: test_data, lstm_nn.labels: test_labels})
        time_stop = time.clock()
        print('Epoch {}: time is {:.1f} sec, error is {:.4f}\n'.format(i + 1, time_stop - time_start, epoch_error))


if __name__ == '__main__':
    main()
