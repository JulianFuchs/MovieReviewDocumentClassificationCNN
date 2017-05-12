import sys
import tensorflow as tf
import math
import Options

import tensorflow as tf


class CNN():

    def __init__(self, options):

        vocab_size = options._voc_size
        max_document_length = options._max_document_length
        max_sentence_length = options._max_sentence_length

        embedding_size = 70

        filter_sizes = (3, 4, 5)
        num_filters = 50
        sentence_rep_size = 150

        document_filter_size = 3
        document_rep_size = 50

        num_classes = 2


        # Placeholders for input, output and dropout
        # input is [None, max_document_length, max_sentence_length]
        # todo: change max_doc_length to be None as well
        self.input_x = tf.placeholder(tf.int32, [None, max_document_length, max_sentence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        '''now for every sentence: extract vector [max_sentence_length], apply embedding on each int representing
        a word. This results in tensor: [word_embedding, max_sentence_length]. Apply CNN on this tensor'''

        # ============================= Embedding Layer =============================

        # define Variable used for embedding
        embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="Embedding")

        embedding = tf.nn.embedding_lookup(embedding_matrix, self.input_x)
        # embedding has shape: [None, max_doc, max_sentence, embedding_size]

        # self.embedding = tf.expand_dims(embedding_temp, -1)

        # ============================= Sentence CNN =============================

        # define Variables used for Sentence CNN
        filter_matrices = []
        filter_biases = []

        for i, filter_size in enumerate(filter_sizes):

            filter_shape = [filter_size, embedding_size, 1, num_filters]
            filter_matrix = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

            filter_matrices.append(filter_matrix)
            filter_biases.append(filter_bias)

        sentence_score_matrix = tf.Variable(tf.truncated_normal([sentence_rep_size, num_classes], stddev=0.1), name="W")
        sentence_score_bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

        sentence_representations = []

        sentence_losses = []

        # iterate over all sentences, apply sentence CNN on each sentence
        for d in range(0, max_document_length):

            # slice(input_, begin, size, name=None)
            sentence = tf.slice(embedding, [0, d, 0, 0], [-1, 1, -1, -1])
            # sentence has shape: [None, 1, max_sentence, embedding_size]
            sentence = tf.squeeze(sentence, 1)
            # sentence has shape: [None, max_sentence, embedding_size]
            sentence = tf.expand_dims(sentence, -1)
            # sentence has shape: [None, max_sentence, embedding_size, 1]

            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                # conv2d: takes input: [batch, in_height, in_width, in_channels] and
                # filter: [filter_height, filter_width, in_channels, out_channels]
                conv = tf.nn.conv2d(
                    sentence,
                    filter_matrices[i],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # conv has shape: [None, max_sentence_length - filter_size + 1, 1, num_filters]

                # Add filter bias, apply non-linearity
                non_lin_vec = tf.nn.relu(tf.nn.bias_add(conv, filter_biases[i]), name="relu")
                # non_lin_vec has shape: [None, max_sentence_length - filter_size + 1, 1, num_filters]

                # Max-pooling over the outputs, takes max of entire feature map, results in 1 int
                pooled = tf.nn.max_pool(
                    non_lin_vec,
                    ksize=[1, max_sentence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # pooled has shape: [None, 1, 1, num_filters]
                pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)

            combined_pools = tf.concat(pooled_outputs, 3)
            # combined_pools has shape: [None, 1, 1, num_filters_total]
            combined_pools = tf.reshape(combined_pools, [-1, num_filters_total])
            # combined pools has shape: [None, num_filters_total]

            sentence_representations.append(combined_pools)

            if sentence_rep_size != num_filters_total:
                print('sentence_rep_size != num_filters_total')

            # target replication via sentence loss
            scores = tf.nn.xw_plus_b(combined_pools, sentence_score_matrix, sentence_score_bias)

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            # losses has shape: [None, ]

            avg_sentence_loss_over_batch = tf.reduce_mean(losses)

            sentence_losses.append(avg_sentence_loss_over_batch)

        doc = tf.stack(sentence_representations, axis=1)
        # doc has shape: [None, max_document_length, sentence_rep_size]
        doc = tf.expand_dims(doc, -1)
        # doc has shape: [None, max_document_length, sentence_rep_size, 1]


        # ============================= Document CNN =============================
        # 50 filters of size document_filter_size

        # Convolution Layer
        filter_shape = [document_filter_size, sentence_rep_size, 1, num_filters]
        filter_matrix = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        filter_bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

        # conv2d: takes input: [batch, in_height, in_width, in_channels] and
        # filter: [filter_height, filter_width, in_channels, out_channels]
        conv = tf.nn.conv2d(
            doc,
            filter_matrix,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        # Apply nonlinearity
        non_lin_vec = tf.nn.relu(tf.nn.bias_add(conv, filter_bias), name="relu")
        # non_lin_vec has shape: [None, max_document_length - document_filter_size + 1, 1, num_filters]

        # Max-pooling over the outputs
        pooled = tf.nn.max_pool(
            non_lin_vec,
            ksize=[1, max_document_length - document_filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        # pooled has shape: [None, 1, 1, document_rep_size=num_filters]

        if document_rep_size != num_filters:
            print('document_rep_size != num_filters')

        pooled = tf.reshape(pooled, [-1, document_rep_size])
        # pooled has shape: [None, document_rep_size]

        # Add dropout
        pooled = tf.nn.dropout(pooled, self.dropout_keep_prob)

        W = tf.Variable(tf.truncated_normal([document_rep_size, num_classes], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

        scores = tf.nn.xw_plus_b(pooled, W, b)
        # scores has shape: [None, num_classes]
        predictions = tf.argmax(scores, axis=1, name="predictions")
        # predictions has shape: [None, ]. A shape of [x, ] means a vector of size x

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
        # losses has shape: [None, ]

        avg_loss = tf.reduce_mean(losses)

        # include target replication, i.e., sentence_losses:
        self._loss = avg_loss + options._lambda_regularizer_strength/len(sentence_losses) * sum(sentence_losses)

        # my optimizer + optimize function
        optimizer = tf.train.AdamOptimizer()
        self._optimize = optimizer.minimize(self._loss)

        correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

if __name__ == '__main__':
    print('building tensor flow model cnn test')

    options = Options.Options()

    cnn = CNN(options)

