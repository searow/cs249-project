#!/bin/bash

# Make the data directory, get the data from the internet, unzip, and save.
# The results are stored in ../data/SCWS.
mkdir -p ../data
curl -J -L http://www-nlp.stanford.edu/~ehhuang/SCWS.zip -o ../data/scws.zip
unzip ../data/scws.zip -d ../data
rm ../data/scws.zip

