import tensorflow as tf
import numpy as np
import DataGenerator
import CNNModel
import math
import time
import sys
import os
import Options

'''
Inspired by https://github.com/scharmchi/char-level-cnn-tf
'''
class Main:
    def __init__(self, options):

        self._options = options

        self._data_generator = DataGenerator.DataGenerator(options)

        self._test_set_offset = 100
        self._valid_set_offset = 0
        self._validation_test_set_size = 100

        self._cnn_model = CNNModel.CNN(options)

        config = tf.ConfigProto(intra_op_parallelism_threads=options._number_of_threads,
                                inter_op_parallelism_threads=options._number_of_threads)

        self._sess = tf.Session(config=config)

        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)

        start = time.time()

        self.evaluate_on_validation()

        for e in range(0, options._epochs):
            print('Starting epoch ' + str(e + 1))

            self.run_epoch()

            print('After finishing epoch ' + str(e + 1))
            self.evaluate_on_validation()

            print('')

        end = time.time()
        print()
        print('Entire program took ' + str((end - start) / 60) + ' minutes')

    def run_epoch(self):
        print('Training model on training set...')

        start_epoch = time.time()
        # take batch_size/2 positive examples

        index = self._valid_set_offset + self._validation_test_set_size
        half_batch_size = math.floor(self._options._batch_size / 2)

        # todo: actually include the final data points as well
        step = 0
        optimize_time_sum = 0

        while index + half_batch_size < 1000:

            batch = self._data_generator.generate_batch(index, half_batch_size)

            pos_labels = np.zeros(shape=(half_batch_size, 2), dtype=float)

            for i in range(0, half_batch_size):
                pos_labels[i, 1] = 1

            neg_labels = np.zeros(shape=(half_batch_size, 2), dtype=float)

            for i in range(0, half_batch_size):
                neg_labels[i, 0] = 1

            labels = np.concatenate((pos_labels, neg_labels), axis=0)

            if step % 5 == 0 and self._options._verbose_mode: #True: #
                acc = self._sess.run(self._cnn_model._accuracy, {self._cnn_model.input_x: batch,
                                                                 self._cnn_model.input_y: labels,
                                                                 self._cnn_model.dropout_keep_prob : 0.5})

                percentage = (step * half_batch_size) / (1000 - self._validation_test_set_size)

                print('Accuracy after ' + str(math.floor(percentage * 100)) + '%: '  + str(acc))
                # if step > 0:
                #     print('On average, an optimize call took: ' + str(optimize_time_sum / step) + ' seconds')

            '''model needs input:
            input_x: [batch_size, char_voc, max_seq, 1]
            input_y: [batch_size, classes]
            '''
            start_optimize = time.time()
            self._sess.run(self._cnn_model._optimize, {self._cnn_model.input_x: batch,
                                                       self._cnn_model.input_y: labels,
                                                       self._cnn_model.dropout_keep_prob: 0.5})
            end_optimize = time.time()

            optimize_time_sum += end_optimize-start_optimize

            index += half_batch_size

            step += 1

        end_epoch = time.time()
        print('Finished training one epoch')
        print('Training took ' + str((end_epoch - start_epoch) / 60) + ' minutes')
        print('On average, an optimize call took: ' + str(optimize_time_sum/step) + ' seconds\n')

    def evaluate_on_validation(self):

        print('Evaluating model on validation set...')
        start = time.time()

        half_batch_size = math.floor(self._options._batch_size / 2)

        accuracies = []

        step = 0
        index = self._valid_set_offset

        while index < self._valid_set_offset + self._validation_test_set_size:

            batch = self._data_generator.generate_batch(index, half_batch_size)

            pos_labels = np.zeros(shape=(half_batch_size, 2), dtype=float)

            for i in range(0, half_batch_size):
                pos_labels[i, 1] = 1

            neg_labels = np.zeros(shape=(half_batch_size, 2), dtype=float)

            for i in range(0, half_batch_size):
                neg_labels[i, 0] = 1

            labels = np.concatenate((pos_labels, neg_labels), axis=0)

            acc = self._sess.run(self._cnn_model._accuracy, {self._cnn_model.input_x: batch,
                                                             self._cnn_model.input_y: labels,
                                                             self._cnn_model.dropout_keep_prob: 0.5})

            accuracies.append(acc)

            if step % 5 == 0 and self._options._verbose_mode:  # True: #
                percentage = (step * half_batch_size) / self._validation_test_set_size
                print('Accuracy after ' + str(math.floor(percentage*100)) + '%: ' + str(acc))

            index += half_batch_size
            step += 1

        end = time.time()
        total_accuracy = sum(accuracies)/len(accuracies)

        print('Evaluating validation set took ' + str((end - start)/60) + ' minutes')
        print('Accuracy over entire validation set: ' + str(math.floor(100*total_accuracy)) + '%')


if __name__ == '__main__':

    options = Options.Options()

    Main(options)

    # if len(sys.argv) != 5:
    #     print('To run model, input: pos_path, neg_path, epochs, num_threads')
    # else:
    #     print('Starting Tweet Sentiment char based CNN analysis \n')
    #     Main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


