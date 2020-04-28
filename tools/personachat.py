import os
from nltk import word_tokenize, pos_tag
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

data_path = 'data/personachat'
resource_path = 'data/personachat/resource'
vocab_path = os.path.join(data_path, 'demo_20000.vocab.pt')

with open(os.path.join(resource_path,'stopwords.txt')) as f:
    stopwords = f.read().split('\n\n')

train = []
flush = False
with open(os.path.join(resource_path, 'train_both_original.txt')) as f:
    data = f.readlines()
    for x in data:
        x = x.strip()
        if x[x.index(' ')+1:x.index(' ')+15] == "your persona: ":
            if not flush:
                flush = True
                pre_dialogue = None
                train.append([[],[]])
                train.append([[],[]])
            train[-2][0].append(x[x.index(' ')+15:])
        elif x[x.index(' ')+1:x.index(' ')+20] == "partner\'s persona: ":
            train[-1][0].append(x[x.index(' ')+20:])
        else:
            dialogue = x[x.index(' ')+1:].split('\t')[:2]
            if '?' not in dialogue[1]:
                train[-2][1].append(dialogue)
            if pre_dialogue and '?' not in dialogue[0]:
                train[-1][1].append([pre_dialogue[1],dialogue[0]])
            flush = False
            pre_dialogue = dialogue


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
            if '?' not in dialogue[1]:
                valid[-2][1].append(dialogue)
            if pre_dialogue and '?' not in dialogue[0]:
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
            if '?' not in dialogue[1]:
                test[-2][1].append(dialogue)
            if pre_dialogue and '?' not in dialogue[0]:
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

sentences = list(sentences.keys())
keywords = []
tagset = ['NOUN','VERB','ADJ','ADV']
for i in range(len(sentences)):
    sentences[i] = [re.sub('\d+','<num>',x) for x in word_tokenize(sentences[i])]

if os.path.isfile(vocab_path):
    vocab_list = torch.load(vocab_path)['src']['itos']
    vocab_dict = dict(zip(vocab_list,list(range(len(vocab_list)))))
else:
    vocab_dict = {}
    for v in sentences:
        for w in v:
            try:
                vocab_dict[w] += 1
            except KeyError:
                vocab_dict[w] = 1
    vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    vocab_list = [x for x,y in vocab_list]
    vocab_list = start_vocab + vocab_list[:20000]
    vocab_dict = dict([(y,x) for x,y in enumerate(vocab_list)])
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
    del embedding_dict
    vocab = {'src':{'itos':vocab_list,'embeddings':embeddings}}
    vocab['tgt'] = vocab['src']
    vocab['cue'] = vocab['src']
    torch.save(vocab, vocab_path)


for i in trange(len(train)):
    train[i][0] = [[BOS_ID]+[vocab_dict.get(y, UNK_ID) for y in sentences[x]]+[EOS_ID] for x in train[i][0]]
    for j in range(len(train[i][1])):
        post = sentences[train[i][1][j][0]]
        response = sentences[train[i][1][j][1]]
        keywords = []
        for x in pos_tag(response, tagset='universal'):
            if x[0] in vocab_list and x[0] not in stopwords and x[1] in tagset:
                keywords.append(x[0])
        post = [vocab_dict.get(x, UNK_ID) for x in post]
        response = [BOS_ID]+[vocab_dict.get(x, UNK_ID) for x in response]+[EOS_ID]
        keywords = [BOS_ID]+[vocab_dict.get(x, UNK_ID) for x in keywords]+[EOS_ID]
        train[i][1][j] = [post,response,keywords]
        
for i in trange(len(valid)):
    valid[i][0] = [[BOS_ID]+[vocab_dict.get(y, UNK_ID) for y in sentences[x]]+[EOS_ID] for x in valid[i][0]]
    for j in range(len(valid[i][1])):
        post = sentences[valid[i][1][j][0]]
        response = sentences[valid[i][1][j][1]]
        keywords = []
        for x in pos_tag(response, tagset='universal'):
            if x[0] in vocab_list and x[0] not in stopwords and x[1] in tagset:
                keywords.append(x[0])
        post = [vocab_dict.get(x, UNK_ID) for x in post]
        response = [BOS_ID]+[vocab_dict.get(x, UNK_ID) for x in response]+[EOS_ID]
        keywords = [BOS_ID]+[vocab_dict.get(x, UNK_ID) for x in keywords]+[EOS_ID]
        valid[i][1][j] = [post,response,keywords]
        
for i in trange(len(test)):
    test[i][0] = [[BOS_ID]+[vocab_dict.get(y, UNK_ID) for y in sentences[x]]+[EOS_ID] for x in test[i][0]]
    for j in range(len(test[i][1])):
        post = sentences[test[i][1][j][0]]
        response = sentences[test[i][1][j][1]]
        keywords = []
        for x in pos_tag(response, tagset='universal'):
            if x[0] in vocab_list and x[0] not in stopwords and x[1] in tagset:
                keywords.append(x[0])
        post = [vocab_dict.get(x, UNK_ID) for x in post]
        response = [BOS_ID]+[vocab_dict.get(x, UNK_ID) for x in response]+[EOS_ID]
        keywords = [BOS_ID]+[vocab_dict.get(x, UNK_ID) for x in keywords]+[EOS_ID]
        test[i][1][j] = [post,response,keywords]

# idf = [0 for _ in range(len(vocab_list))]
# data = train + valid + test
# knowledge = [y[1:-1] for x in data for y in x[0]]
# for x in knowledge:
#     for y in set(x):
#         idf[y] += 1
# idf = np.log(len(knowledge)/(1+np.array(idf)))
#
# index = []
# for i in range(len(train)):
#     index.append([])
#     knowledge = [x[1:-1] for x in train[i][0]]
#     tf = np.zeros((len(knowledge),len(vocab_list)))
#     for j in range(len(knowledge)):
#         for x in knowledge[j]:
#             tf[j,x] += 1
#     tf /= np.sum(tf,axis=1,keepdims=True)
#     tf_idf = tf*idf[np.newaxis,:]
#     keywords = np.argsort(-tf_idf,axis=1)[:,:2]
#     for j in range(len(train[i][1])):
#         index[-1].append([])
#         response = train[i][1][j][1][1:-1]
#         for k in range(len(knowledge)):
#             if keywords[k,0] in response or keywords[k,1] in response:
#                 index[i][j].append(k)

data = {}
data['train'] = []
for x in train:
    for y in x[1]:
        data['train'].append({'src':y[0],'tgt':y[1],'cue':x[0],'keywords':y[2]})
data['valid'] = []
for x in valid:
    for y in x[1]:
        data['valid'].append({'src':y[0],'tgt':y[1],'cue':x[0],'keywords':y[2]})
data['test'] = []
for x in test:
    for y in x[1]:
        data['test'].append({'src':y[0],'tgt':y[1],'cue':x[0],'keywords':y[2]})
torch.save(data, os.path.join(data_path, 'demo_20000.data.pt'))
