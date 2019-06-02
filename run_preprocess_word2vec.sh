#!/bin/bash

python='python3'

CUDA_VISIBLE_DEVICES=$1 ${python} preprocess.py \
--type word2vec \
--vector_path data/word2vec/sgns.sogounews.bigram-char \
--train_file data/sinanews.train \
--test_file data/sinanews.test \
--output_path data/word2vec_temp
