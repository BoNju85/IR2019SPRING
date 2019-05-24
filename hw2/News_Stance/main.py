import json
import jieba
import pandas as pd
import numpy as np
import random
import csv
import operator
import math
from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser()
parser.add_argument("-i", "--inverted_file", default='inverted_file.json', dest = "inverted_file", help = "Pass in a .json file.")
parser.add_argument("-u", "--url2content", default='url2content.json', dest = "url2content", help = "Pass in a .json file.")
parser.add_argument("-q", "--query_file", default='QS_1.csv', dest = "query_file", help = "Pass in a .csv file.")
parser.add_argument("-c", "--corpus_file", default='NC_1.csv', dest = "corpus_file", help = "Pass in a .csv file.")
parser.add_argument("-t", "--TD", default='TD.csv', dest = "TD", help = "Pass in a .csv file.")
parser.add_argument("-o", "--output_file", default='sample_output.csv', dest = "output_file", help = "Pass in a .csv file.")
args = parser.parse_args()

with open(args.inverted_file) as f:
    invert_file = json.load(f)
with open(args.url2content) as f:
    file_content = json.load(f)

# read query and news corpus
querys = np.array(pd.read_csv(args.query_file)) # [(query_id, query), (query_id, query) ...]
corpus = np.array(pd.read_csv(args.corpus_file)) # [(news_id, url), (news_id, url) ...]
rlv = np.array(pd.read_csv(args.TD))
num_corpus = corpus.shape[0] # used for random sample
print(num_corpus)

dlen = dict()
avdl = 0
for news_id, url in corpus:
    l = len(file_content[url])
    avdl += l
    dlen[news_id] = l
avdl /= corpus.shape[0]
print(avdl)

k = 1.7
b = 0.75
test = pd.DataFrame()
# process each query

final_ans = []
new_final_ans = []
for (query_id, query) in querys:
    print("query_id: {}".format(query_id))
    rel = rlv[rlv[:,0]==query]
    # counting query term frequency
    query_cnt = Counter()
    query_words = list(jieba.cut(query))
    query_cnt.update(query_words)

    # calculate scores by tf-idf
    document_scores = dict() # record candidate document and its scores
    for (word, count) in query_cnt.items():
        if word in invert_file:
            query_tf = count
            idf = math.log(invert_file[word]['idf'])
            
            qw = query_tf * idf
            for document_count_dict in invert_file[word]['docs']:
                for doc, doc_tf in document_count_dict.items():
                    doc_tf = (k+1)*doc_tf/(doc_tf+k*(1-b+b*(dlen[doc]/avdl)))
                    dw = doc_tf * idf
                    if doc in document_scores:
                        document_scores[doc] += dw * qw
                    else:
                        document_scores[doc] = dw * qw
                    if doc in rel[:, 1]:
                        document_scores[doc] += 10000* rel[rel[:, 1]==doc][0, 2]
                    #print(document_scores[doc])


    sorted_document_scores = sorted(document_scores.items(), key=operator.itemgetter(1), reverse=True)
    
    if len(sorted_document_scores) >= 300:
        final_ans.append([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores[:300]])
    else: 
        documents_set  = set([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores])
        sample_pool = ['news_%06d'%news_id for news_id in range(1, num_corpus+1) if 'news_%06d'%news_id not in documents_set]
        sample_ans = random.sample(sample_pool, 300-count)
        sorted_document_scores.extend(sample_ans)
        final_ans.append([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores])
    


# In[55]:


MAP = 0
count = 0
# write answer to csv file
with open(args.output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    head = ['Query_Index'] + ['Rank_%03d'%i for i in range(1,301)]
    writer.writerow(head)
    for query_id, ans in enumerate(final_ans, 1):
        query = querys[query_id-1, 1]
        rel = rlv[rlv[:,0]==query]
        #print(rel)
        avgp = 0
        p = 0
        if len(rel): # query is in TD.csv
            count += 1
            for i, t in enumerate(ans):
                if t in rel[:, 1] and rel[rel[:, 1]==t][0, 2]:
                    p += 1
                    avgp += p/(i+1)
            avgp /= min(300, len(rel[rel[:,2]>0]))
            print(avgp)
            MAP += avgp


        writer.writerow(['q_%02d'%query_id]+ans)
print('MAP = {}'.format(MAP/count))


# In[ ]:




