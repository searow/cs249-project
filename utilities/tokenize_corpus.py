#!/usr/bin/env python3

# Usage:
# $ python tokenize_corpus.py <file_to_tokenize> <vocab_size> <output_file_name>

# The <file_to_tokenize> should be space delimited words, no punctuation,
# same as in the text8 corpus for word2vec example.
# Output file will be in the same directory as the directory that contains
# <file_to_tokenize>.

# Modified from word2vec_basic.py from tensorflow.

import os
import sys
import tensorflow as tf
import collections
import pickle

if len(sys.argv) != 4:
    print('Incorrect number of input arguments. Use the following:')
    print('$ python tokenize_corpus.py <file_to_tokenize> <vocab_size> <output_file_name>')
    sys.exit(1)

# Get the full path to the desired file, exit with RC=1 if failure.
full_file_path = os.path.realpath(sys.argv[1])
if not os.path.isfile(full_file_path):
    print('Unable to find the file provided: ' + full_file_path)
    print('Exiting')
    sys.exit(1)

# Parse the input vocab size argument, exit if can't be an int.
try:
    vocabulary_size = int(sys.argv[2])
except ValueError:
    print('Input vocab size was not an int')
    print('Exiting')
    sys.exit(1)

# Get the final full file path for the output file.
input_file_dir = os.path.split(full_file_path)[0]
output_file = sys.argv[3]
output_file_path = os.path.join(input_file_dir, output_file)

# Print input arguments for debugging.
print('Input file: ' + full_file_path)
print('Vocab size: ' + str(vocabulary_size))
print('Output file: ' + output_file_path)

# Read corpus and get as list of words.
print('Reading corpus...')
with open(full_file_path, 'r') as readfile:
    vocabulary = tf.compat.as_str(readfile.read()).split()

# Print some stats.
print('Corpus has ' + str(len(vocabulary)) + ' words')

# Tokenize the corpus.
print('Tokenizing corpus...')
count = [['UNK', -1]]
count.extend(collections.Counter(vocabulary).most_common(vocabulary_size - 1))
dictionary = dict()
for word, _ in count:
    dictionary[word] = len(dictionary)
data = list()
unk_count = 0
for word in vocabulary:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
        unk_count += 1
    data.append(index)
count[0][1] = unk_count
reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

# Save the data into the output data file.
print('Saving tokenized corpus...')
save_obj = {
    'data': data,
    'count': count,
    'dictionary': dictionary,
    'reversed_dictionary': reversed_dictionary
}
with open(output_file_path, 'wb') as writefile:
    pickle.dump(save_obj, writefile)
