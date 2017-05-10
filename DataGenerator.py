import tensorflow as tf
import numpy as np
import random
import os
import sys
import operator
import Options


class DataGenerator():
    def __init__(self, options):
        print('Initializing DataGenerator, generating vocabulary')

        self._options = options

        self._voc_count_dict = {}

        self.analyze_dataset()

        self.generate_vocabulary()

        sorted_voc_count_dict = sorted(self._voc_count_dict.items(), key=operator.itemgetter(1))

        print('The entire vocabulary has size: ' + str(len(sorted_voc_count_dict)))

        # reverse the list so that most encountered word is first
        list.reverse(sorted_voc_count_dict)

        self._vocabulary = {}

        # 0: special symbol for padding. 1: special symbol for unknown word
        self._int_for_padding = 0
        self._int_for_unknown = 1


        for i in range(0, options._voc_size-2):
            self._vocabulary[sorted_voc_count_dict[i][0]] = i + 2


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


    '''Takes half_batch_size documents from pos and neg files each and combines them into 1 batch
    The output is: [batch_size, _max_document_length, _max_sentence_length]'''
    def generate_batch(self, start_ind, half_batch_size):

        batch = np.ndarray(shape=(2*half_batch_size,
                                  self._options._max_document_length,
                                  self._options._max_sentence_length),
                                    dtype=float)  # Creates a 3D array with uninitialized values

        files = self._options._pos_files[start_ind : start_ind + half_batch_size]
        files = files + self._options._neg_files[start_ind : start_ind + half_batch_size]

        batch_it = 0

        for file in files:

            document = open(file, 'r').read()

            sentences = document.split('\n')

            for s_i, s in enumerate(sentences):
                words = s.split(' ')

                for w_i, w in enumerate(words):
                    if len(w) > 1:
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

            batch_it += 1

        return batch

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

        print('The longest document has length : ' + str(longest_document_length))
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

    batch = data_generator.generate_batch(0, 10)
