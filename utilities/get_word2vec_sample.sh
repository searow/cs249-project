#!/bin/bash

# Copied from https://github.com/tensorflow/models/tree/master/tutorials/embedding

# Gets the data at the URL, unzips it, and puts it in the data directory.
mkdir -p ../data
mkdir -p ../data/word2vec_sample

curl http://mattmahoney.net/dc/text8.zip > text8.zip
unzip text8.zip -d ../data/word2vec_sample
curl https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip > source-archive.zip
unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > ../data/word2vec_sample/questions-words.txt
rm text8.zip source-archive.zip

