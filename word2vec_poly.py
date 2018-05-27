#!/usr/bin/env python

# Usage:
# $ python word2vec_poly.py -h

# Modified from word2vec_basic.py from tensorflow.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import pickle
import numpy as np
import datetime
import time
import pytz

import collections
import random
import tensorflow as tf
import math

# Separate annoying tensorflow import warnings.
print('--- End warnings ---\n\n\n')

# Get the current timestamp for saving a unique fileid.
ts = datetime.datetime.now(pytz.timezone('US/Pacific'))
timestamp = ts.strftime('%Y-%m-%d-%H:%M:%S')

# Parse the command line arguments into the FLAGS variable.
parser = argparse.ArgumentParser()
parser.add_argument('--corpus',
                    type=str,
                    required=True,
                    help='The path to the tokenized corpus to train on')
parser.add_argument('--log_dir',
                    type=str,
                    required=True,
                    help='The log directory to store TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Make sure that the corpus file exists before continuing.
if not os.path.isfile(FLAGS.corpus):
    print('The corpus file could not be found')
    sys.exit(1)

# Make the log directory if it does not already exist.
if not os.path.exists(FLAGS.log_dir):
    print('Specified log directory does not exist')
    print('Creating log directory...')
    os.makedirs(FLAGS.log_dir)

# Import tokenized data, count, and dictionary.
print('Reading the tokenized corpus...')
with open(FLAGS.corpus, 'rb') as readfile:
    read_obj = pickle.load(readfile)
    data = read_obj['data']
    count = read_obj['count']
    dictionary = read_obj['dictionary']
    reversed_dictionary = read_obj['reversed_dictionary']
    vocabulary_size = len(count)
# data is the tokenized corpus as a list of tokens [2, 5, 2, 1, 6, etc]
# count is a list of tuples with each tuple representing (word, count)
# dictionary is a dictionary of {word: token_number}
# reversed_dictionary is a dictionary of {token_number: word}
# vocabulary_size is how many unique words we want to keep track of, all other
#     words get mapped to 0 (UNKNOWN)


# Step 3: Function to generate a training batch for the skip-gram model.
# data_index is a global used to keep track of index between calls.
data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    """Generates a batch from data object

    Generates a new batch at every call. Only uses source words that have the
    entire context window available on both sides. Given the current starting
    index, defines the skip_window on both sides of the current word, and then
    randomly chooses num_skips number of training examples from the window,
    without replacement. If num_skips = 2 * skip_window, then all the
    context words inside of the skip_window will be used exactly once.

    Args:
        batch_size:  Int, how many training examples in the batch
        num_skips:   Int, how many words you want to predict for each input.
                     If the current word is dog, and the sentence is:
                     [and the fast dog runs home today], then num_skips is how
                     many words we want to predict from dog. If we have
                     num_skips = 2, then each input word (dog) will produce
                     2 training examples for this batch. So we might produce
                     (dog, fast) and (dog, today) as our input if num_skips = 2.

        skip_window: Int, the context window from which to create training
                     examples. This is a double-sided window, so one word
                     before and one word after would correspond to window = 1.
    Returns:
        batch:  A 1-D array of size (batch_size), each element corresponding
                to an input token, the current word (input to the skip-gram).
        labels: A 2-D array of size (batch_size, 1), each element corresponding
                to the correct context word for this training example. For our
                case, we only have 1 target word, giving us (batch_size, 1).
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

# Model parameters
batch_size = 128
num_embeddings = 3 # How many embeddings per word.
embedding_size = 300 // num_embeddings  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.


# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# Build the computational graph.
print('Building computational graph...')
graph = tf.Graph()
with graph.as_default():

    # Input data, these are pretty much just (word, context) pairs and the
    # validation data indices.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Make one variable for each polysemous embedding and include their
        # embedding mapping as well.
        embeddings = [None] * num_embeddings
        embeds = [None] * num_embeddings
        for i in range(num_embeddings):
            with tf.name_scope('embeddings-' + str(i)):
                embeddings[i] = tf.Variable(
                    tf.random_uniform(
                        [vocabulary_size, embedding_size], -1.0, 1.0))
                embeds[i] = tf.nn.embedding_lookup(embeddings[i], train_inputs)

        # Make one variable for each polysemous embedding's weight matrix.
        nce_weights = [None] * num_embeddings
        for i in range(num_embeddings):
            with tf.name_scope('weights-' + str(i)):
                nce_weights[i] = tf.Variable(
                    tf.truncated_normal(
                        [vocabulary_size, embedding_size],
                        stddev=1.0 / math.sqrt(embedding_size)))

        # Make one bias for each polysemous embedding's weight matrix.
        nce_biases = [None] * num_embeddings
        for i in range(num_embeddings):
            with tf.name_scope('biases-' + str(i)):
                nce_biases[i] = tf.Variable(tf.zeros([vocabulary_size]))

    # Create the loss functions for each embeddings, of which there are emb^2.
    losses = [[None] * num_embeddings] * num_embeddings
    loss_stack = [None] * (num_embeddings ** 2)
    # Loss functions for each embeddings and loss functions for each weight.
    stack_counter = 0
    for i in range(num_embeddings): # Embeddings
        for j in range(num_embeddings): # Weights
            with tf.name_scope('losses-' + str(i) + '-' + str(j)):
                losses[i][j] = tf.nn.nce_loss(
                    weights=nce_weights[j],
                    biases=nce_biases[j],
                    labels=train_labels,
                    inputs=embeds[i],
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size)
            # # Add the loss value as a scalar to summary.
            # tf.summary.scalar('losses-' + str(i) + '-' + str(j), losses[i][j])

            # Add the loss to the stack of losses.
            loss_stack[stack_counter] = losses[i][j]
            stack_counter += 1

    # loss_stack is a list of the loss tensors, so we want to reshape that
    # into a (batch_size, num_embeddings^2) shape matrix.
    with tf.name_scope('individual-losses'):
        loss_grouped = tf.stack(loss_stack, axis=1)

    # Add the MIN node at the end of the loss functions.
    with tf.name_scope('min-losses'):
        loss_min = tf.reduce_min(loss_grouped, axis=1)

    # Add the MEAN function at the end of it all for the batch.
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(loss_min)

    # Add the global loss to the summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute normalized verison of the embeddings.
    normalized_embeddings = [None] * num_embeddings
    for i in range(num_embeddings):
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings[i]), 1, keep_dims=True))
        normalized_embeddings[i] = embeddings[i] / norm


    embed_stack = tf.stack(normalized_embeddings, axis=0)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

# Begin training.
num_steps = 100001
with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

    # Initialize the embeddings, weights, and biases.
    print('Initializing variables...')
    init.run()

    print('Training...')
    # Actual training steps.
    average_loss = 0
    for step in range(num_steps):
        # Generate a batch of inputs for this step.
        batch_inputs, batch_labels = generate_batch(batch_size,
                                                    num_skips,
                                                    skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # Define metadata variable (?).
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
        # Feed metadata variable to session for visualizing the graph in TensorBoard.
        _, summary, loss_val = session.run([optimizer, merged, loss],
                                           feed_dict=feed_dict,
                                           run_metadata=run_metadata)
        average_loss += loss_val

        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)

        # Add metadata to visualize the graph for the last run.
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)

        # Print average loss every 2000 steps.
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # average_loss is an estimate over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Save the normalized embeddings every n steps.
        if step % 2000 == 0:
            save_embedding = embed_stack.eval()
            fn = 'saved_embeddings_step_{}_k_{}_dim_{}_{}'.format(step, num_embeddings, embedding_size, timestamp)
            np.save(fn, save_embedding)
            print(fn, save_embedding.shape)

        # # Perform evaluation (slow) every 10000 steps.
        # if step % 10000 == 0:
        #     sim = similarity.eval()
        #     for i in xrange(valid_size):
        #         valid_word = reverse_dictionary[valid_examples[i]]
        #         top_k = 8  # number of nearest neighbors
        #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        #         log_str = 'Nearest to %s:' % valid_word
        #         for k in xrange(top_k):
        #             close_word = reverse_dictionary[nearest[k]]
        #             log_str = '%s %s,' % (log_str, close_word)
        #         print(log_str)

    # final_embeddings = normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
    with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
        for i in range(vocabulary_size):
            f.write(reversed_dictionary[i] + '\n')

    # Save the model for checkpoints.
    saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt')) 

writer.close()
