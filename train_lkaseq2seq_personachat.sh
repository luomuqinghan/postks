#!/bin/bash
python ./network.py --model lkaseq2seq --data_dir ./data/personachat --save_dir ./models/lkaseq2seq/personachat --embed_file /home/cx/WordEmbedding/glove.840B.300d.txt --max_vocab_size 20000 --lr 0.0005 --max_dec_len 21 --valid_steps 958 --gpu 2