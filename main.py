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

        self._cnn_model = CNNModel.CNN(options)#LinearModel.LinearModel(options)#

        config = tf.ConfigProto(intra_op_parallelism_threads=options._number_of_threads,
                                inter_op_parallelism_threads=options._number_of_threads)

        self._sess = tf.Session(config=config)

        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)

        start = time.time()

        print('Sanity test, running model without training on validation set:')
        self.evaluate_on_validation()
        print('')

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

        self._data_generator.randomize_training_sets()

        index = 0
        half_batch_size = math.floor(self._options._batch_size / 2)

        # todo: actually include the final data points as well
        step = 0
        optimize_time_sum = 0

        losses = []

        while index + half_batch_size <= len(self._data_generator._training_set_pos):

            batch, labels = self._data_generator.generate_training_batch(index, half_batch_size)

            if len(batch) == 0 or len(labels) == 0:
                print('batch or labels is empty')

            # set dropout_keep_prob to 1 when evaluating
            if step % 50 == 0 and self._options._verbose_mode: #True: #
                acc = self._sess.run(self._cnn_model._accuracy, {self._cnn_model.input_x: batch,
                                                                 self._cnn_model.input_y: labels,
                                                                 self._cnn_model.dropout_keep_prob : 1})

                percentage = (step * half_batch_size) / (1000 - self._validation_test_set_size)

                print('Accuracy after ' + str(math.floor(percentage * 100)) + '%: ' + str(acc))
                # if step > 0:
                #     print('On average, an optimize call took: ' + str(optimize_time_sum / step) + ' seconds')

            '''model needs input:
            input_x: [batch_size, char_voc, max_seq, 1]
            input_y: [batch_size, classes]
            '''

            loss = self._sess.run(self._cnn_model._loss, {self._cnn_model.input_x: batch,
                                                       self._cnn_model.input_y: labels,
                                                       self._cnn_model.dropout_keep_prob: 0.5})

            losses.append(loss)
            #print(loss)

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
        print('On average, an optimize call took: ' + str(optimize_time_sum/step) + ' seconds')
        print('Average loss: ' + str(sum(losses)/len(losses)))
        print('')

    def evaluate_on_validation(self):

        print('Evaluating model on validation set...')
        start = time.time()

        half_batch_size = math.floor(self._options._batch_size / 2)

        accuracies = []

        step = 0
        index = 0

        while index + half_batch_size <= self._validation_test_set_size/2:

            batch, labels = self._data_generator.generate_validation_batch(index, half_batch_size)

            # for x in range(0, self._options._batch_size):
            #     for y in range(0, self._options._max_document_length):
            #         for z in range(0, self._options._max_sentence_length):
            #             if batch[x][y][z] < 0:
            #                 print('neg value at: [' + str(x) + ', ' + str(y) + ', ' + str(z) + ']')

            # set dropout_keep_prob to 1 when evaluating
            acc = self._sess.run(self._cnn_model._accuracy, {self._cnn_model.input_x: batch,
                                                             self._cnn_model.input_y: labels,
                                                             self._cnn_model.dropout_keep_prob: 1})

            accuracies.append(acc)

            if step % 50 == 0 and self._options._verbose_mode:  # True: #
                percentage = (step * half_batch_size) / self._validation_test_set_size
                print('Accuracy after ' + str(math.floor(percentage*100)) + '%: ' + str(acc))

            index += half_batch_size
            step += 1

        end = time.time()
        total_accuracy = sum(accuracies)/len(accuracies)

        print('Evaluating validation set took ' + str((end - start)/60) + ' minutes')
        print('Accuracy over entire validation set: ' + str(math.floor(100*total_accuracy)) + '%')



def load_options(options_path):
    options_file = open(options_path, 'r').read()

    options_lines = options_file.split('\n')

    voc_size = int(options_lines[0])
    max_document_length = int(options_lines[1])
    max_sentence_length = int(options_lines[2])
    lambda_regularizer_strength = int(options_lines[3])
    data_folder_path = options_lines[4]
    number_of_threads = int(options_lines[5])
    epochs = int(options_lines[6])
    batch_size = int(options_lines[7])
    validation_test_set_size = int(options_lines[8])
    verbose_mode = int(options_lines[9])

    options = Options.Options(voc_size=voc_size,
                              max_document_length=max_document_length,
                              max_sentence_length=max_sentence_length,
                              lambda_regularizer_strength=lambda_regularizer_strength,
                              data_folder_path=data_folder_path,
                              number_of_threads=number_of_threads,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_test_set_size=validation_test_set_size,
                              verbose_mode=verbose_mode)

    return options


if __name__ == '__main__':

    # reading options file
    curr_path = os.path.dirname(os.path.realpath(__file__))

    original_stdout = sys.stdout

    if not os.path.exists(curr_path + '/outputs'):
        os.makedirs(curr_path + '/outputs')

    for options_file_name in os.listdir(curr_path + '/options/'):
        print('Running options file: ' + str(options_file_name))
        options = load_options(str(curr_path) + '/options/' + str(options_file_name))
        output_file = open(curr_path + '/outputs/' + str(options_file_name), 'w')
        sys.stdout = output_file

        Main(options)

        sys.stdout = original_stdout

    # print('Options are: ')
    # print(voc_size)
    # print(max_document_length)
    # print(max_sentence_length)
    # print(lambda_regularizer_strength)
    # print(data_folder_path)
    # print(number_of_threads)
    # print(epochs)
    # print(batch_size)
    # print(validation_test_set_size)
    # print(verbose_mode)
    # print()
    #
    # Main(options)



