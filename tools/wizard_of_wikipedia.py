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
data_path = '../data/wizard_of_wikipedia'
resource_path = '../data/wizard_of_wikipedia/resource'
untokenized_path = '../data/wizard_of_wikipedia/untokenized'
vocab_path = os.path.join(data_path, 'demo_20000.vocab.pt')

data = []
types = ['train', 'valid_seen', 'valid_unseen', 'test_seen', 'test_unseen']
for n in range(len(types)):
    data.append([])
    rawdata = json.load(open(os.path.join(resource_path, types[n]+'.json')))

    print('parsering sentences for %s'%types[n])
    sentences = {}
    for i in trange(len(rawdata)):
        for j in range(len(rawdata[i]['dialog'])):
            if rawdata[i]['dialog'][j]['speaker'].split('_')[1]=='Apprentice':
                for k in range(len(rawdata[i]['dialog'][j]['retrieved_passages'])):
                    for a,b in rawdata[i]['dialog'][j]['retrieved_passages'][k].items():
                        rawdata[i]['dialog'][j]['retrieved_passages'][k] = b
                        for m in range(len(rawdata[i]['dialog'][j]['retrieved_passages'][k])):
                            try:
                                rawdata[i]['dialog'][j]['retrieved_passages'][k][m] = sentences[rawdata[i]['dialog'][j]['retrieved_passages'][k][m]]
                            except KeyError:
                                sentences[rawdata[i]['dialog'][j]['retrieved_passages'][k][m]] = len(sentences)
                                rawdata[i]['dialog'][j]['retrieved_passages'][k][m] = len(sentences)-1

            try:
                rawdata[i]['dialog'][j]['text'] = sentences[rawdata[i]['dialog'][j]['text']]
            except KeyError:
                sentences[rawdata[i]['dialog'][j]['text']] = len(sentences)
                rawdata[i]['dialog'][j]['text'] = len(sentences)-1

    sentences = dict([(v,k) for k,v in sentences.items()])
    print('tokenizing for %s' % types[n])
    for i in trange(len(sentences)):
        sentences[i] = [re.sub('\d+','<num>',x) for x in nltk.word_tokenize(sentences[i].lower())]

    for i in range(len(rawdata)):
        for j in range(len(rawdata[i]['dialog'])):
            if j>=1 and rawdata[i]['dialog'][j]['speaker'].split('_')[1]=='Wizard' and rawdata[i]['dialog'][j-1]['speaker'].split('_')[1]=='Apprentice':
                data[n].append([[],[]])
                knowledge = rawdata[i]['dialog'][j-1]['retrieved_passages']
                knowledge = [y for x in knowledge for y in x]
                knowledge = [sentences[x] for x in knowledge]
                data[n][-1][0] = knowledge
                post = sentences[rawdata[i]['dialog'][j-1]['text']]
                response = sentences[rawdata[i]['dialog'][j]['text']]
                data[n][-1][1].append([post, response])

    data[n] = list(filter(lambda x:len(x[0])>0,data[n]))
    with open(os.path.join(untokenized_path,types[n]+'.json'),'w') as f:
        f.write('\n'.join([json.dumps(x) for x in data[n]]))

train = []
with open(os.path.join(untokenized_path,'train.json')) as f:
    for line in f:
        train.append(json.loads(line))

valid_seen = []
with open(os.path.join(untokenized_path,'valid_seen.json')) as f:
    for line in f:
        valid_seen.append(json.loads(line))

valid_unseen = []
with open(os.path.join(untokenized_path,'valid_unseen.json')) as f:
    for line in f:
        valid_unseen.append(json.loads(line))

test_seen = []
with open(os.path.join(untokenized_path,'test_seen.json')) as f:
    for line in f:
        test_seen.append(json.loads(line))

test_unseen = []
with open(os.path.join(untokenized_path,'test_unseen.json')) as f:
    for line in f:
        test_unseen.append(json.loads(line))

data = train + valid_seen + valid_unseen + test_seen + test_unseen
valid = valid_seen
if os.path.isfile(vocab_path):
    vocab_list = torch.load(vocab_path)['src']['itos']
    vocab_dict = dict(zip(vocab_list, list(range(len(vocab_list)))))
else:
    vocab_dict = {}
    text = [y for x in data for y in x[0]] + [z for x in data for y in x[1] for z in y]
    for s in text:
        for w in s:
            if w in vocab_dict:
                vocab_dict[w] += 1
            else:
                vocab_dict[w] = 1
    del text
    vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    vocab_list = [x for x, y in vocab_list]
    vocab_list = start_vocab + vocab_list[:20000]
    vocab_dict = dict([(y, x) for x, y in enumerate(vocab_list)])

    embeddings = np.zeros((len(vocab_list), 300))
    with open('/home/cx/WordEmbedding/glove.840B.300d.txt') as f:
        weights = [line.strip().split() for line in f.readlines()]
        weights = dict([(' '.join(line[:-300]),np.array([float(x) for x in line[-300:]])) for line in weights])
        for i,word in enumerate(vocab_list):
            try:
                embeddings[i] = weights[word]
            except KeyError:
                pass
    vocab = {'src': {'itos': vocab_list, 'embeddings': embeddings}}
    vocab['tgt'] = vocab['src']
    vocab['cue'] = vocab['src']
    torch.save(vocab, vocab_path)

for i in range(len(train)):
    train[i][0] = [[BOS_ID]+[vocab_dict.get(y, UNK_ID) for y in x]+[EOS_ID] for x in train[i][0]]
    train[i][1] = [[[BOS_ID]+[vocab_dict.get(z, UNK_ID) for z in y]+[EOS_ID] for y in x] for x in train[i][1]]

for i in range(len(valid)):
    valid[i][0] = [[BOS_ID]+[vocab_dict.get(y, UNK_ID) for y in x]+[EOS_ID] for x in valid[i][0]]
    valid[i][1] = [[[BOS_ID]+[vocab_dict.get(z, UNK_ID) for z in y]+[EOS_ID] for y in x] for x in valid[i][1]]

for i in range(len(test_seen)):
    test_seen[i][0] = [[BOS_ID]+[vocab_dict.get(y, UNK_ID) for y in x]+[EOS_ID] for x in test_seen[i][0]]
    test_seen[i][1] = [[[BOS_ID]+[vocab_dict.get(z, UNK_ID) for z in y]+[EOS_ID] for y in x] for x in test_seen[i][1]]

for i in range(len(test_unseen)):
    test_unseen[i][0] = [[BOS_ID]+[vocab_dict.get(y, UNK_ID) for y in x]+[EOS_ID] for x in test_unseen[i][0]]
    test_unseen[i][1] = [[[BOS_ID]+[vocab_dict.get(z, UNK_ID) for z in y]+[EOS_ID] for y in x] for x in test_unseen[i][1]]

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
for x in test_seen:
    for y in x[1]:
        data['test'].append({'src':y[0],'tgt':y[1],'cue':x[0]})
torch.save(data, os.path.join(data_path, 'seen_20000.data.pt'))

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
for x in test_unseen:
    for y in x[1]:
        data['test'].append({'src':y[0],'tgt':y[1],'cue':x[0]})
torch.save(data, os.path.join(data_path, 'unseen_20000.data.pt'))