import os
import sys

class Options:
    def __init__(self, voc_size=10000,
                 max_document_length=113,
                 max_sentence_length=180,
                 data_folder_path='/home/theearlymiddleages/Datasets/MovieDocumentSentimentClassification/',
                 number_of_threads=8,
                 epochs=3,
                 batch_size = 10,#10,
                 verbose_mode=False
                 ):

        self._number_of_threads = number_of_threads
        self._epochs = epochs
        self._batch_size = batch_size
        self._verbose_mode = verbose_mode


        self._voc_size = voc_size
        self._max_document_length = max_document_length
        self._max_sentence_length = max_sentence_length
        self._data_folder_path = data_folder_path

        pos_path = data_folder_path + 'pos/'
        neg_path = data_folder_path + 'neg/'

        self._pos_files = []
        self._neg_files = []

        for pos_file_name in os.listdir(pos_path):
            self._pos_files.append(pos_path + pos_file_name)

        for neg_file_name in os.listdir(neg_path):
            self._neg_files.append(neg_path + neg_file_name)


        # todo: randomize _pos_files/neg_files here (listdir has arbitrary order already though...)