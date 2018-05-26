#!/bin/bash

# Make the data directory, get the data from the internet, unzip, and save.
# The results are stored in ../data/SCWS.
mkdir -p ../data
mkdir -p ../data/SCWS
curl -J -L http://nlp.stanford.edu/data/WestburyLab.wikicorp.201004.txt.bz2 -o ../data/SCWS/WestburyLab.wikicorp.201004.txt.bz2
bunzip2 ../data/SCWS/WestburyLab.wikicorp.201004.txt.bz2 -d

