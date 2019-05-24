import os
import numpy as np
import pandas as pd
import re
import time
import math
import jieba
from scipy.sparse import lil_matrix
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--query-file", dest="fin", default="queries/query-test.xml")
parser.add_argument("-o", "--ranked-list", dest="fout", default="submission.csv")
parser.add_argument("-m", "--model-dir", dest="modeldir", default="model")
parser.add_argument("-d", "--NTCIR-dir", dest="NTCIRdir", default="CIRB010")
parser.add_argument("-r", action='store_true', default=False)
parser.add_argument("-b", action='store_true', default=True)

args = parser.parse_args()


file_id = list()  # id to path
with open(os.path.join(args.modeldir, 'file-list')) as f:
    for line, doc in enumerate(f):
        file_id.append(doc.split('/')[-1].strip().lower())
file_num = len(file_id)
print(len(file_id), file_id[0])

file_len = dict()  # path to len
for root, dirs, files in os.walk(args.NTCIRdir):
    for file in files:
        path = root + '/' + file
        with open(path) as f:
            content = f.read()
            content = re.sub(r'<\/*[a-z]*>', '', content)
            tmp = list(filter(None, content))
            #print(len(tmp))
            file_len[file.lower()] = len(content)
avdl = sum(file_len.values())/file_num
print('Average Doc Length: {}'.format(avdl))


vocab = dict()
with open(os.path.join(args.modeldir, 'vocab.all')) as vfile:
    vfile.readline()
    for line, word in enumerate(vfile):
        vocab[int(line)+1] = word.strip()

start = time.time()
inverted = dict()  # word to count
vid = 0
with open(os.path.join(args.modeldir, 'inverted-file')) as f:
    for line in f:
        vid1, vid2, n = line.strip().split()
        vid1 = int(vid1)
        vid2 = int(vid2)
        if vid2 != -1:
            word = vocab[vid1]+vocab[vid2]
        else:
            word = vocab[vid1]
        Count = dict()  # DocId to count of word
        for i in range(int(n)):
            DocID, count = f.readline().strip().split()
            Count[int(DocID)] = int(count)
            #print(DocID, freq[int(DocID)])
        inverted[word] = Count
        vid += 1

#df_ans = pd.read_csv('queries/ans_train.csv')

k = 1.2
k3 = 500
b = 0.75
related = list()
query_id = list()

with open(args.fin) as f:
    content = f.read()
    for qid, sub in enumerate(content.split('<topic>')[1:]):
        sub = re.sub(r'\n', '', sub)
        n, t, q, nar, concepts = list(filter(None, re.split(r'<\/*[a-z]*>', sub)))
        concepts = re.split(r'、|。', concepts)[:-1]
        
        query_id.append(n[-3:])
        
        for concept in concepts:
            if len(concept) > 2:
                for i in range(len(concept)-1):
                    concepts.append(concept[i:i+2])
    
        dv = np.zeros((file_num, len(concepts)))  # doc score
        qv = np.zeros(len(concepts))
        for i, word in enumerate(concepts):
            if len(word) > 2:
                continue
            try:
                df = sum(inverted[word].values())
                IDF = math.log((file_num-df+0.5)/(df+0.5))
                qv[i] += IDF
                for occur, count in inverted[word].items():
                    length = file_len[file_id[occur]]
                    TF = (k+1)*count/(count+k*(1-b+b*length/avdl))
                    dv[occur][i] += TF * IDF
            except:
                continue
        sim = np.sum(dv, 1)
        sim[np.isnan(sim)] = 0
        rid = [x for x in np.argsort(sim)[::-1][:100]]
        irid = [x for x in np.argsort(sim)[::-1][100:200]]
        topK = [file_id[x] for x in rid]
        bottomK = [file_id[x] for x in irid]
        
        if args.r:
            upgrade_num = 3
            alpha = 0.8
            beta = 0.1
            gamma = 0.1
            for i in range(upgrade_num):
                qv = alpha*qv + beta*np.sum([dv[x] for x in rid], 0)/100 - gamma*np.sum([dv[x] for x in irid], 0)/100
                sim = np.dot(dv, qv)
                sim[np.isnan(sim)] = 0

                rid = [x for x in np.argsort(sim)[::-1][:100]]
                irid = [x for x in np.argsort(sim)[::-1][100:200]]
                topK = [file_id[x] for x in rid]
                bottomK = [file_id[x] for x in irid]
        
        related.append(' '.join(topK))


df_ans = pd.DataFrame({'query_id': query_id, 'retrieved_docs': related})

df_ans.to_csv(args.fout, index=False)




