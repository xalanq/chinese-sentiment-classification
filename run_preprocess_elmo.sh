#!/bin/bash

python='python3'

CUDA_VISIBLE_DEVICES=$1 ${python} preprocess.py \
--type elmo \
--elmo_model_path data/zhs.model \
--train_file data/sinanews.train \
--test_file data/sinanews.test \
--output_path data/elmo_temp
