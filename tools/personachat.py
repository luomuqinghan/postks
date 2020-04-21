import os
import sys
import json
import numpy as np
import re
import torch
from tqdm import trange
from copy import deepcopy
import spacy

nlp = spacy.load('en_core_web_sm')
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

start_vocab = ['<pad>', '<unk>', '<bos>', '<eos>']

data_path = 'data/personachat'
resource_path = 'data/personachat/resource'
vocab_path = os.path.join(data_path, 'demo_20000.vocab.pt')

train_file = os.path.join(data_path, 'train.json')
valid_file = os.path.join(data_path, 'valid.json')
test_file = os.path.join(data_path, 'test.json')
if os.path.exists(train_file) and os.path.exists(valid_file) and os.path.exists(test_file):
    train = []
    with open(os.path.join(data_path, 'train.json')) as f:
        for line in f:
            train.append(json.loads(line))
    valid = []
    with open(os.path.join(data_path, 'valid.json')) as f:
        for line in f:
            valid.append(json.loads(line))
    test = []
    with open(os.path.join(data_path, 'test.json')) as f:
        for line in f:
            test.append(json.loads(line))
else:
    train = []
    flush = False
    with open(os.path.join(resource_path, 'train_both_original.txt')) as f:
        data = f.readlines()
        for x in data:
            x = x.strip()
            if x[x.index(' ') + 1:x.index(' ') + 15] == "your persona: ":
                if not flush:
                    flush = True
                    pre_dialogue = None
                    train.append([[], []])
                    train.append([[], []])
                train[-2][0].append(x[x.index(' ') + 15:])
            elif x[x.index(' ') + 1:x.index(' ') + 20] == "partner\'s persona: ":
                train[-1][0].append(x[x.index(' ') + 20:])
            else:
                dialogue = x[x.index(' ') + 1:].split('\t')[:2]
                train[-2][1].append(dialogue)
                if pre_dialogue:
                    train[-1][1].append([pre_dialogue[1], dialogue[0]])
                flush = False
                pre_dialogue = dialogue

    valid = []
    flush = False
    with open(os.path.join(resource_path, 'valid_both_original.txt')) as f:
        data = f.readlines()
        for x in data:
            x = x.strip()
            if x[x.index(' ') + 1:x.index(' ') + 15] == "your persona: ":
                if not flush:
                    flush = True
                    pre_dialogue = None
                    valid.append([[], []])
                    valid.append([[], []])
                valid[-2][0].append(x[x.index(' ') + 15:])
            elif x[x.index(' ') + 1:x.index(' ') + 20] == "partner\'s persona: ":
                valid[-1][0].append(x[x.index(' ') + 20:])
            else:
                dialogue = x[x.index(' ') + 1:].split('\t')[:2]
                valid[-2][1].append(dialogue)
                if pre_dialogue:
                    valid[-1][1].append([pre_dialogue[1], dialogue[0]])
                flush = False
                pre_dialogue = dialogue
    test = []
    flush = False
    with open(os.path.join(resource_path, 'test_both_original.txt')) as f:
        data = f.readlines()
        for x in data:
            x = x.strip()
            if x[x.index(' ') + 1:x.index(' ') + 15] == "your persona: ":
                if not flush:
                    flush = True
                    pre_dialogue = None
                    test.append([[], []])
                    test.append([[], []])
                test[-2][0].append(x[x.index(' ') + 15:])
            elif x[x.index(' ') + 1:x.index(' ') + 20] == "partner\'s persona: ":
                test[-1][0].append(x[x.index(' ') + 20:])
            else:
                dialogue = x[x.index(' ') + 1:].split('\t')[:2]
                test[-2][1].append(dialogue)
                if pre_dialogue:
                    test[-1][1].append([pre_dialogue[1], dialogue[0]])
                flush = False
                pre_dialogue = dialogue

    with open(train_file, 'w') as f:
        f.write('\n'.join([json.dumps(x) for x in train]))

    with open(valid_file, 'w') as f:
        f.write('\n'.join([json.dumps(x) for x in valid]))

    with open(test_file, 'w') as f:
        f.write('\n'.join([json.dumps(x) for x in test]))

data = train + valid + test
sentences = [y for x in data for y in x[0]] + [z for x in data for y in x[1] for z in y]
sentences = list(set(sentences))
sentences = dict(list(zip(sentences, list(range(len(sentences))))))

for i in range(len(train)):
    train[i][0] = [sentences[x] for x in train[i][0]]
    train[i][1] = [[sentences[y] for y in x] for x in train[i][1]]

for i in range(len(valid)):
    valid[i][0] = [sentences[x] for x in valid[i][0]]
    valid[i][1] = [[sentences[y] for y in x] for x in valid[i][1]]

for i in range(len(test)):
    test[i][0] = [sentences[x] for x in test[i][0]]
    test[i][1] = [[sentences[y] for y in x] for x in test[i][1]]

sentences = list(sentences.keys())
if os.path.exists(os.path.join(data_path,'word_tokenized.pt')) and os.path.join(data_path,'dependency_parsed.pt'):
    word_tokenized = torch.load(os.path.join(data_path, 'word_tokenized.pt'))
    dependency_parsed = torch.load(os.path.join(data_path, 'dependency_parsed.pt'))
else:
    word_tokenized = []
    dependency_parsed = []
    print('word tokenizing and dependency parsing')
    sys.stdout.flush()
    i = 0
    for doc in nlp.pipe(sentences, disable=["tagger", "ner"]):
        i += 1
        if i%100==0:
            print(i,len(sentences))
        word_tokenized.append([re.sub('\d+', '<num>', x.text) for x in doc])
        dependency_parsed.append([(x.dep_,x.i,x.head.i) for x in doc])
    torch.save(word_tokenized,os.path.join(data_path,'word_tokenized.pt'))
    torch.save(dependency_parsed,os.path.join(data_path,'dependency_parsed.pt'))

if os.path.isfile(vocab_path):
    vocab_list = torch.load(vocab_path)['src']['itos']
    dependency_label = torch.load(vocab_path)['src']['dependency_label']
else:
    vocab_dict = {}
    for v in word_tokenized:
        for w in v:
            try:
                vocab_dict[w] += 1
            except KeyError:
                vocab_dict[w] = 1
    vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    vocab_list = [x for x,y in filter(lambda x:x[1]>1,vocab_list)]
    vocab_list = start_vocab + vocab_list[:20000]
    embeddings = np.zeros((len(vocab_list), 300))
    embedding_dict = {}
    with open('/home/cx/WordEmbedding/glove.6B.300d.txt') as f:
        lines = f.read()
    lines = lines.split('\n')
    for i in trange(len(lines)):
        lines[i] = lines[i].split()
        embedding_dict[' '.join(lines[i][:-300])] = np.array([float(x) for x in lines[i][-300:]])
    del lines
    for i in range(len(vocab_list)):
        try:
            embeddings[i] = embedding_dict[vocab_list[i]]
        except KeyError:
            pass

    dependency_label = list(set([y[0] for x in dependency_parsed for y in x]))
    dependency_label.remove('ROOT')
    dependency_label = ['none','self']+['forward_'+x for x in dependency_label]+['backward_'+x for x in dependency_label]

    vocab = {'src': {'itos': vocab_list, 'embeddings': embeddings, 'dependency_label':dependency_label}}
    vocab['tgt'] = vocab['src']
    vocab['cue'] = vocab['src']
    torch.save(vocab, vocab_path)

dependency_label = dict(list(zip(dependency_label, list(range(len(dependency_label))))))
vocab_dict = dict([(y, x) for x, y in enumerate(vocab_list)])
for i in range(len(dependency_parsed)):
    dependency = [(1,j,j) for j in range(len(dependency_parsed[i]))]
    for x in dependency_parsed[i]:
        if x[0] != 'ROOT':
            dependency.append((dependency_label['forward_'+x[0]],x[1]-1,x[2]-1))
            dependency.append((dependency_label['backward_'+x[0]],x[2]-1,x[1]-1))
    dependency_array = np.identity(len(dependency_parsed[i]),dtype=np.int64)
    for x in dependency:
        dependency_array[x[1], x[2]] = x[0]
    dependency_parsed[i] = dependency_array.tolist()

train_word_tokenized = deepcopy(train)
train_dependency_parsed = deepcopy(train)
for i in range(len(train_word_tokenized)):
    train_word_tokenized[i][0] = [[BOS_ID] + [vocab_dict.get(y, UNK_ID) for y in word_tokenized[x]] + [EOS_ID] for x in train_word_tokenized[i][0]]
    train_word_tokenized[i][1] = [[[BOS_ID] + [vocab_dict.get(z, UNK_ID) for z in word_tokenized[y]] + [EOS_ID] for y in x] for x in train_word_tokenized[i][1]]
for i in range(len(train_dependency_parsed)):
    train_dependency_parsed[i][0] = [dependency_parsed[x] for x in train_dependency_parsed[i][0]]
    train_dependency_parsed[i][1] = [[dependency_parsed[y] for y in x] for x in train_dependency_parsed[i][1]]

valid_word_tokenized = deepcopy(valid)
valid_dependency_parsed = deepcopy(valid)
for i in range(len(valid_word_tokenized)):
    valid_word_tokenized[i][0] = [[BOS_ID] + [vocab_dict.get(y, UNK_ID) for y in word_tokenized[x]] + [EOS_ID] for x in valid_word_tokenized[i][0]]
    valid_word_tokenized[i][1] = [[[BOS_ID] + [vocab_dict.get(z, UNK_ID) for z in word_tokenized[y]] + [EOS_ID] for y in x] for x in valid_word_tokenized[i][1]]
for i in range(len(valid_dependency_parsed)):
    valid_dependency_parsed[i][0] = [dependency_parsed[x] for x in valid_dependency_parsed[i][0]]
    valid_dependency_parsed[i][1] = [[dependency_parsed[y] for y in x] for x in valid_dependency_parsed[i][1]]

test_word_tokenized = deepcopy(test)
test_dependency_parsed = deepcopy(test)
for i in range(len(test_word_tokenized)):
    test_word_tokenized[i][0] = [[BOS_ID] + [vocab_dict.get(y, UNK_ID) for y in word_tokenized[x]] + [EOS_ID] for x in test_word_tokenized[i][0]]
    test_word_tokenized[i][1] = [[[BOS_ID] + [vocab_dict.get(z, UNK_ID) for z in word_tokenized[y]] + [EOS_ID] for y in x] for x in test_word_tokenized[i][1]]
for i in range(len(test_dependency_parsed)):
    test_dependency_parsed[i][0] = [dependency_parsed[x] for x in test_dependency_parsed[i][0]]
    test_dependency_parsed[i][1] = [[dependency_parsed[y] for y in x] for x in test_dependency_parsed[i][1]]

data = {}

data['train'] = []
for x,y in zip(train_word_tokenized,train_dependency_parsed):
    word_tokenized = {}
    dependency_parsed = {}
    for i in range(len(x[1])):
        sample = {}
        sample['src'] = x[1][i][0]
        sample['tgt'] = x[1][i][1]
        sample['cue'] = x[0]
        sample['src_dependency'] = y[1][i][0]
        sample['tgt_dependency'] = y[1][i][1]
        sample['cue_dependency'] = y[0]
        data['train'].append(sample)
data['valid'] = []
for x,y in zip(valid_word_tokenized,valid_dependency_parsed):
    word_tokenized = {}
    dependency_parsed = {}
    for i in range(len(x[1])):
        sample = {}
        sample['src'] = x[1][i][0]
        sample['tgt'] = x[1][i][1]
        sample['cue'] = x[0]
        sample['src_dependency'] = y[1][i][0]
        sample['tgt_dependency'] = y[1][i][1]
        sample['cue_dependency'] = y[0]
        data['valid'].append(sample)
data['test'] = []
for x,y in zip(test_word_tokenized,test_dependency_parsed):
    word_tokenized = {}
    dependency_parsed = {}
    for i in range(len(x[1])):
        sample = {}
        sample['src'] = x[1][i][0]
        sample['tgt'] = x[1][i][1]
        sample['cue'] = x[0]
        sample['src_dependency'] = y[1][i][0]
        sample['tgt_dependency'] = y[1][i][1]
        sample['cue_dependency'] = y[0]
        data['test'].append(sample)
torch.save(data, os.path.join(data_path, 'demo_20000.data.pt'))