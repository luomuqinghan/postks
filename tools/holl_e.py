import os
import nltk
import json
import numpy as np
import re
import torch
from tqdm import trange

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

start_vocab = ['<pad>', '<unk>', '<bos>', '<eos>']

data_path = 'data/Holl-E/'
resource_path = 'data/Holl-E/resource/main_data'
vocab_path = os.path.join(data_path, 'demo_20000.vocab.pt')

train = []
flush = False
with open(os.path.join(resource_path, 'train_data.json')) as f:
    data = json.load(f)

valid = []
flush = False
with open(os.path.join(resource_path, 'valid_both_original.txt')) as f:
    data = f.readlines()
    for x in data:
        x = x.strip()
        if x[x.index(' ')+1:x.index(' ')+15] == "your persona: ":
            if not flush:
                flush = True
                pre_dialogue = None
                valid.append([[],[]])
                valid.append([[],[]])
            valid[-2][0].append(x[x.index(' ')+15:])
        elif x[x.index(' ')+1:x.index(' ')+20] == "partner\'s persona: ":
            valid[-1][0].append(x[x.index(' ')+20:])
        else:
            dialogue = x[x.index(' ')+1:].split('\t')[:2]
            valid[-2][1].append(dialogue)
            if pre_dialogue:
                valid[-1][1].append([pre_dialogue[1],dialogue[0]])
            flush = False
            pre_dialogue = dialogue

test = []
flush = False
with open(os.path.join(resource_path, 'test_both_original.txt')) as f:
    data = f.readlines()
    for x in data:
        x = x.strip()
        if x[x.index(' ')+1:x.index(' ')+15] == "your persona: ":
            if not flush:
                flush = True
                pre_dialogue = None
                test.append([[],[]])
                test.append([[],[]])
            test[-2][0].append(x[x.index(' ')+15:])
        elif x[x.index(' ')+1:x.index(' ')+20] == "partner\'s persona: ":
            test[-1][0].append(x[x.index(' ')+20:])
        else:
            dialogue = x[x.index(' ')+1:].split('\t')[:2]
            test[-2][1].append(dialogue)
            if pre_dialogue:
                test[-1][1].append([pre_dialogue[1],dialogue[0]])
            flush = False
            pre_dialogue = dialogue

with open(os.path.join(data_path,'train.json'),'w') as f:
    f.write('\n'.join([json.dumps(x) for x in train]))

with open(os.path.join(data_path,'valid.json'),'w') as f:
    f.write('\n'.join([json.dumps(x) for x in valid]))

with open(os.path.join(data_path,'test.json'),'w') as f:
    f.write('\n'.join([json.dumps(x) for x in test]))

data = train + valid + test
sentences = [y for x in data for y in x[0]] + [z for x in data for y in x[1] for z in y]
sentences = list(set(sentences))
sentences = dict(list(zip(sentences,list(range(len(sentences))))))

for i in range(len(train)):
    train[i][0] = [sentences[x] for x in train[i][0]]
    train[i][1] = [[sentences[y] for y in x] for x in train[i][1]]

for i in range(len(valid)):
    valid[i][0] = [sentences[x] for x in valid[i][0]]
    valid[i][1] = [[sentences[y] for y in x] for x in valid[i][1]]

for i in range(len(test)):
    test[i][0] = [sentences[x] for x in test[i][0]]
    test[i][1] = [[sentences[y] for y in x] for x in test[i][1]]

sentences = dict([(y,x) for x,y in sentences.items()])

for k,v in sentences.items():
    sentences[k] = [re.sub('\d+','<num>',x) for x in nltk.word_tokenize(v)]

if os.path.isfile(vocab_path):
    vocab_list = torch.load(vocab_path)['src']['itos']
    vocab_dict = dict(zip(vocab_list,list(range(len(vocab_list)))))
else:
    vocab_dict = {}
    for k,v in sentences.items():
        for w in v:
            try:
                vocab_dict[w] += 1
            except KeyError:
                vocab_dict[w] = 1
    vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    vocab_list = [x for x,y in vocab_list]
    vocab_list = start_vocab + vocab_list[:20000]
    vocab_dict = dict([(y,x) for x,y in enumerate(vocab_list)])
    embeddings = np.zeros((len(vocab_list),300))
    with open('/home/cx/WordEmbedding/glove.6B.300d.txt') as f:
        lines = f.readlines()
        for i in trange(len(lines)):
            line = lines[i].strip().split()
            weight = np.array([float(x) for x in line[-300:]])
            word = ' '.join(line[:-300])
            if word in vocab_list:
                embeddings[vocab_list.index(word)] = weight
    vocab = {'src':{'itos':vocab_list,'embeddings':embeddings}}
    vocab['tgt'] = vocab['src']
    vocab['cue'] = vocab['src']
    torch.save(vocab, vocab_path)

for i in range(len(train)):
    train[i][0] = [[BOS_ID]+[vocab_dict.get(y, UNK_ID) for y in sentences[x]]+[EOS_ID] for x in train[i][0]]
    train[i][1] = [[[BOS_ID]+[vocab_dict.get(z, UNK_ID) for z in sentences[y]]+[EOS_ID] for y in x] for x in train[i][1]]

for i in range(len(valid)):
    valid[i][0] = [[BOS_ID]+[vocab_dict.get(y, UNK_ID) for y in sentences[x]]+[EOS_ID] for x in valid[i][0]]
    valid[i][1] = [[[BOS_ID]+[vocab_dict.get(z, UNK_ID) for z in sentences[y]]+[EOS_ID] for y in x] for x in valid[i][1]]

for i in range(len(test)):
    test[i][0] = [[BOS_ID]+[vocab_dict.get(y, UNK_ID) for y in sentences[x]]+[EOS_ID] for x in test[i][0]]
    test[i][1] = [[[BOS_ID]+[vocab_dict.get(z, UNK_ID) for z in sentences[y]]+[EOS_ID] for y in x] for x in test[i][1]]

data = {}
data['train'] = []
for x in train:
    for y in x[1]:
        data['train'].append({'src':y[0],'tgt':y[1],'cue':x[0]})
data['valid'] = []
for x in valid:
    for y in x[1]:
        data['valid'].append({'src':y[0],'tgt':y[1],'cue':x[0]})
data['test'] = []
for x in test:
    for y in x[1]:
        data['test'].append({'src':y[0],'tgt':y[1],'cue':x[0]})
torch.save(data, os.path.join(data_path, 'demo_20000.data.pt'))