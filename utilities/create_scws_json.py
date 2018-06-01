#!/usr/bin/env python3

# Reads test dataset for SCWS task and puts the data into pickle format for
# easier usage with python.
# Run the utility within the same directory as the file is kept:
# $ ./create-scws-json.py

import csv
import os
import pickle
import string
import re

# Relative path of the directory, ratings file, and output file.
rel_path = '../data/SCWS'
ratings_file = 'ratings.txt'
output_file = 'ratings.pickle'

# Holds the final data to save as pickle.
output_data = []

# def clean_line(line):
#     """Cleans a string of text by trying to emulate clean_data.pl."""
#     # Replace all numbers first.
#     line = line.replace("0", " zero ")
#     line = line.replace("1", " one ")
#     line = line.replace("2", " two ")
#     line = line.replace("3", " three ")
#     line = line.replace("4", " four ")
#     line = line.replace("5", " five ")
#     line = line.replace("6", " six ")
#     line = line.replace("7", " seven ")
#     line = line.replace("8", " eight ")
#     line = line.replace("9", " nine ")
#     # Remove all non-ascii characters and replace with spaces.
#     for punct in string.punctuation:
#         if punct == '<' or punct == '>':
#             continue
#         line = line.replace(punct, " ")

#     assert(line.count('<b>') == 1)
#     assert(line.count('</b>') == 1)

#     # Transform all spaces into single spaces. The tokenization script
#     # already does split(), but this might be useful for consistency later.
#     for i in range(10):
#         line = line.replace("  ", " ")
#         line = line.replace("   ", " ")
#         line = line.replace("    ", " ")
#         line = line.replace("     ", " ")

#     # Make all letters lower case.
#     line = line.lower()

#     # Replace the <b> </b> tags first since every line will have one of them.
#     line = line.replace("<b>", "")
#     line = line.replace("</b>", "")

#     return line


def clean_line(line, input_word):
    """Cleans a string of text by trying to emulate clean_data.pl."""
    if line.count('<b>') != 1:
        print('***', line, line.count('<b>'))
    assert(line.count('<b>') == 1)
    assert(line.count('</b>') == 1)

    # Transform all spaces into single spaces. The tokenization script
    # already does split(), but this might be useful for consistency later.
    for i in range(10):
        line = line.replace("  ", " ")
        line = line.replace("   ", " ")
        line = line.replace("    ", " ")
        line = line.replace("     ", " ")

    # Make all letters lower case.
    line = line.lower()


    # Remove all non-ascii characters and replace with spaces.
    words = line.split()
    for i in range(len(words) - 1, -1, -1):
        word = words[i]
        if word == input_word.lower():
            continue
        orig_word = word
        word = word.replace("0", " zero ")
        word = word.replace("1", " one ")
        word = word.replace("2", " two ")
        word = word.replace("3", " three ")
        word = word.replace("4", " four ")
        word = word.replace("5", " five ")
        word = word.replace("6", " six ")
        word = word.replace("7", " seven ")
        word = word.replace("8", " eight ")
        word = word.replace("9", " nine ")
        if orig_word != word:
            del words[i]
            for j, new_word in enumerate(word.split()):
                words.insert(i + j, new_word)
        delete = False
        for p in string.punctuation:
            if p in word:
                if word == '<b>' or word == '</b>':
                    continue
                delete = True
        if delete:
            del words[i]

    sb_ind = None
    eb_ind = None

    # print(words)
    for i, word in enumerate(words):
        if word == '<b>':
            sb_ind = i
        elif word == '</b>':
            eb_ind = i
    # print(sb_ind, eb_ind)
    assert(sb_ind is not None and eb_ind is not None)

    del words[eb_ind]
    del words[sb_ind]

    if words[sb_ind].lower() != input_word.lower():
        print(line)
        print(words)
        print(sb_ind)
        print(input_word)
    assert(words[sb_ind] == input_word.lower())

    return ' '.join(words), sb_ind

# Open the raw file in TSV format.
with open(os.path.join(rel_path, ratings_file), 'r') as tsvfile:
    # tsvreader = csv.reader(tsvfile, delimiter='\t')


    # Populate the output data by reading each line and the corresponding tab.
    for line in tsvfile:
        line = line.rstrip().split('\t')
        line_1, ind_1 = clean_line(line[5], line[1])
        line_2, ind_2 = clean_line(line[6], line[3])
        # for x in line[7]:
        #     try:
        #         float(x)
        #     except:
        #         print('^^^', x, '**', line)
        data = {
            'id': int(line[0]),
            'word1': line[1],
            'POS_of_word1': line[2],
            'word2': line[3],
            'POS_of_word2': line[4],
            'word1_in_context': line_1,
            'word2_in_context': line_2,
            'word1_index': int(ind_1),
            'word2_index': int(ind_2),
            'average_human_rating': float(line[7])
        }

        # Each human rating is a separate tab at the end.
        data['10_individual_human_ratings'] = []
        for i in range(8, 18):
            data['10_individual_human_ratings'].append(float(line[i]))


        # Put into final object.
        output_data.append(data)

# Save the data as a pickle file in the same directory as original file.
with open(os.path.join(rel_path, output_file), 'wb') as picklefile:
    pickle.dump(output_data, picklefile)

