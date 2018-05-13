#!/usr/bin/env python3

# Reads test dataset for SCWS task and puts the data into JSON format for
# easier usage with python.
# Run the utility within the same directory as the file is kept:
# $ ./create-scws-json.py

import json
import csv
import os

# Relative path of the directory, ratings file, and output file.
rel_path = '../data/SCWS'
ratings_file = 'ratings.txt'
output_file = 'ratings.json'

# Holds the final data to save as json.
output_data = []

# Open the raw file in TSV format.
with open(os.path.join(rel_path, ratings_file), 'r') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter='\t')

    # Populate the output data by reading each line and the corresponding tab.
    for line in tsvreader:
        data = {
            'id': line[0],
            'word1': line[1],
            'POS_of_word1': line[2],
            'word2': line[3],
            'POS_of_word2': line[4],
            'word1_in_context': line[5],
            'word2_in_context': line[6],
            'average_human_rating': line[7],
        }

        # Each human rating is a separate tab at the end.
        data['10_individual_human_ratings'] = []
        for i in range(8, 18):
            data['10_individual_human_ratings'].append(line[i])


        # Put into final object.
        output_data.append(data)

# Save the data as a json file in the same directory as original file.
with open(os.path.join(rel_path, output_file), 'w') as jsonfile:
    json.dump(output_data, jsonfile)

