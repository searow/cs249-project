#!/bin/bash

# Starts docker container on host port 8888, mounts the current directory,
# which should be called from the project root directory, into the container
# at /notebooks/cs249.
docker run -it -p 8888:8888 -p 6006:6006 -v "$PWD:/notebooks/cs249" searow/cs249

