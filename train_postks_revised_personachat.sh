#!/bin/bash
python ./network.py --gpu 3 --use_gs True --gs_tau 0.2 --model postks --data_dir ./data/revised_personachat --save_dir ./models/postks/revised_personachat --embed_file /home/cx/WordEmbedding/glove.6B.300d.txt --max_vocab_size 20000 --lr 0.0005 --max_dec_len 21 --valid_steps 958