#!/bin/bash
python ./network.py --gpu 0 --use_gs True --gs_tau 0.2 --model postks --data_dir ./data/personachat --max_dec_len 21 --max_vocab_size 20000 --use_posterior False --gen_file ./output/postks/personachat/test.result --gold_score_file ./output/postks/personachat/gold.scores --ckpt ./models/postks/personachat/best.model --test > output/postks/personachat/test.log
