import tensorflow as tf
import numpy as np
import random
import os
import sys
import operator
import Options
import math


class DataGenerator():
    def __init__(self, options):
        print('Initializing DataGenerator, generating vocabulary')

        self._options = options

        self._voc_count_dict = {}

        self.analyze_dataset()

        self._total_words_count = 0
        self.generate_vocabulary()

        sorted_voc_count_dict = sorted(self._voc_count_dict.items(), key=operator.itemgetter(1))

        print('The entire vocabulary has size: ' + str(len(sorted_voc_count_dict)))

        # reverse the list so that most encountered word is first
        list.reverse(sorted_voc_count_dict)

        self._vocabulary = {}

        # 0: special symbol for padding. 1: special symbol for unknown word
        self._int_for_padding = 0
        self._int_for_unknown = 1

        vocabulary_word_coverage = 0
        for i in range(0, options._voc_size - 2):
            self._vocabulary[sorted_voc_count_dict[i][0]] = i + 2
            vocabulary_word_coverage += sorted_voc_count_dict[i][1]

        print('With a vocabulary size of ' + str(options._voc_size) + ', ' +
              str(math.floor(vocabulary_word_coverage/self._total_words_count*100)) + '% of all words are covered')

        random.shuffle(options._pos_files)
        random.shuffle(options._neg_files)

        split_index = math.floor(options._validation_test_set_size/2)

        self._test_validation_set_pos = options._pos_files[0:split_index]
        self._test_validation_set_neg = options._neg_files[0:split_index]

        self._training_set_pos = options._pos_files[split_index:options._number_of_files_per_class]
        self._training_set_neg = options._neg_files[split_index:options._number_of_files_per_class]

    '''reads all documents and generates entire vocabulary'''
    def generate_vocabulary(self):

        for file in self._options._pos_files + self._options._neg_files:

            document = open(file, 'r').read()

            sentences = document.split('\n')

            for s in sentences:
                words = s.split(' ')

                for w in words:
                    if len(w) > 1:
                        if w in self._voc_count_dict:
                            self._voc_count_dict[w] += 1
                        else:
                            self._voc_count_dict[w] = 1

                        self._total_words_count += 1


    def generate_validation_batch(self, start_index, half_batch_size):
        return self.generate_batch(self._test_validation_set_pos[start_index:start_index + half_batch_size],
                                    self._test_validation_set_neg[start_index:start_index + half_batch_size])

    def generate_training_batch(self, start_index, half_batch_size):
        return self.generate_batch(self._training_set_pos[start_index:start_index + half_batch_size],
                                    self._training_set_neg[start_index:start_index + half_batch_size])

    def randomize_training_sets(self):
        random.shuffle(self._training_set_pos)
        random.shuffle(self._training_set_neg)

    '''Takes half_batch_size documents from pos and neg files each and combines them into 1 batch
    The output is: [batch_size, _max_document_length, _max_sentence_length]'''
    def generate_batch(self, pos_files, neg_files):

        if len(pos_files) != len(neg_files):
            print('pos_file not same length as neg_file')

        batch = np.ndarray(shape=(len(pos_files) + len(neg_files),
                                  self._options._max_document_length,
                                  self._options._max_sentence_length),
                                    dtype=float)

        # for testing purposes, so that we're sure we set every value in batch
        batch.fill(-1)

        batch_it = 0
        for file in pos_files + neg_files:

            document = open(file, 'r').read()

            sentences = document.split('\n')

            # set s_i to -1 initially, because if there is an empty document, s_i will never be set to anything in
            # enumerate(sentences), which means every entry in the batch will be the default value
            s_i = -1

            for s_i, s in enumerate(sentences):
                words = s.split(' ')

                words = [w for w in words if len(w) > 1]

                # set w_i to -1 initially, because if words is empty, w_i will never be set at all in enumerate(words).
                # this will result in the entire sentence representation being the default value
                w_i = -1

                if s_i < self._options._max_document_length:
                    for w_i, w in enumerate(words):
                        if w_i < self._options._max_sentence_length:
                            if w in self._vocabulary:
                                batch[batch_it][s_i][w_i] = self._vocabulary[w]
                            else:
                                # insert unknown token
                                batch[batch_it][s_i][w_i] = self._int_for_unknown

                    w_i += 1

                    while w_i < self._options._max_sentence_length:
                        batch[batch_it][s_i][w_i] = self._int_for_padding
                        w_i += 1

            s_i += 1

            while s_i < self._options._max_document_length:
                for w_i in range(0, self._options._max_sentence_length):
                    batch[batch_it][s_i][w_i] = self._int_for_padding
                s_i += 1

            # for y in range(0, self._options._max_document_length):
            #     for z in range(0, self._options._max_sentence_length):
            #         if batch[batch_it][y][z] < 0:
            #             print('neg value at: [' + str(batch_it) + ', ' + str(y) + ', ' + str(z) + ']')

            batch_it += 1

        pos_labels = np.zeros(shape=(len(pos_files), 2), dtype=float)
        neg_labels = np.zeros(shape=(len(pos_files), 2), dtype=float)

        for i in range(0, len(pos_files)):
            pos_labels[i, 1] = 1
            neg_labels[i, 0] = 1

        labels = np.concatenate((pos_labels, neg_labels), axis=0)

        return batch, labels

    '''analyzes entire dataset, finds max_sentence_length and max_doc_length'''
    def analyze_dataset(self):

        total_document_length = 0
        total_documents = 0

        longest_document_length = 0
        longest_document_name = ''

        longest_sentence_length = 0
        longest_sentence = ''

        total_sentence_length = 0
        total_sentences = 0

        for file in self._options._pos_files + self._options._neg_files:

            document = open(file, 'r').read()
            sentences = document.split('\n')

            total_document_length += len(sentences)
            total_documents += 1

            if len(sentences) > longest_document_length:
                longest_document_length = len(sentences)
                longest_document_name = file

            for s in sentences:
                words = s.split(' ')
                total_sentence_length += len(words)
                total_sentences += 1

                if len(words) > longest_sentence_length:
                    longest_sentence_length = len(words)
                    longest_sentence = s


        print('Average document length: ' + str(total_document_length / total_documents))
        print('Average sentence length: ' + str(total_sentence_length / total_sentences))
        print()

        print('The longest document has length: ' + str(longest_document_length))
        print('The longest sentence has length: ' + str(longest_sentence_length))
        print()


        # print('The longest document has length : ' + str(self._longest_document_length) +
        #       ' and is found at: ' + self._longest_document_name)
        #
        # print('The longest sentence has length: ' + str(self._longest_sentence_length) +
        #         ' and is : ' + self._longest_sentence)


if __name__ == '__main__':
    print('DataGenerator Test')

    options = Options.Options()

    data_generator = DataGenerator(options)

    batch = data_generator.generate_validation_batch(0, 10)
