#!/bin/bash

# Gets the data at the URL, unzips it, and puts it in the data directory.
mkdir -p ../data
mkdir -p ../data/enwik9

curl http://mattmahoney.net/dc/enwik9.zip > enwik9.zip
unzip enwik9.zip -d ../data/enwik9
