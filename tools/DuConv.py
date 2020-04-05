import os
import json
import numpy as np
import torch

rawdata_path = 'rawdata/DuConv'
data_path = 'data/DuConv'

data = torch.load(os.path.join(rawdata_path,'demo_30000.data.pt'))

train = []
for x in data['train']:
    train.append([])
    train[-1].append([y[1:-1] for y in x['cue']])
    train[-1].append([x['src'][1:-1]])
    train[-1].append([x['tgt'][1:-1]])
open(os.path.join(data_path,'train.json'),'w').write(json.dumps(train))

valid = []
for x in data['valid']:
    valid.append([])
    valid[-1].append([y[1:-1] for y in x['cue']])
    valid[-1].append([x['src'][1:-1]])
    valid[-1].append([x['tgt'][1:-1]])
open(os.path.join(data_path,'valid.json'),'w').write(json.dumps(valid))

test = []
for x in data['test']:
    test.append([])
    test[-1].append([y[1:-1] for y in x['cue']])
    test[-1].append([x['src'][1:-1]])
    test[-1].append([x['tgt'][1:-1]])
open(os.path.join(data_path,'test.json'),'w').write(json.dumps(test))

vocab = torch.load(os.path.join(rawdata_path,'demo_30000.vocab.pt'))
vocab = vocab['src']['itos']
open(os.path.join(data_path,'vocab.txt'),'w').write('\n'.join(vocab))