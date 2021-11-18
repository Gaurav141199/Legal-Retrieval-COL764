from bs4 import BeautifulSoup
import math
import lxml
import os
import json
import re
import snappy
from PorterStemmer import PorterStemmer
import numpy as np
import pandas as pd
import nltk
import rank_bm25
import pickle
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
print(stop_words)

directory= r"AILA_2019_dataset/Object_casedocs/"

def stemmer(tokenizedText):
    ps=PorterStemmer()
    stemmedText=[]
    for word in tokenizedText:
        stemmedText.append(ps.stem(word.lower(),0,len(word)-1))
    return stemmedText

def tokenizer(docText):
    tokenizedText=re.split('/| |;|,|\*|\n|:|\(|\)|`|\.|\{|\}|\'|\"|\[|\]|\_|>|^|<|\t|=|#',docText)
    tokenizedText=set(tokenizedText)
    if '' in tokenizedText:
        tokenizedText.remove('')
    ret = tokenizedText.copy()
    for e in tokenizedText:
        if e.lower() in stopwords:
            ret.remove(e)
    return ret
    
def tokenizerForDocs(docText):
    tokenizedText=re.split('/| |;|,|\*|\n|:|\(|\)|`|\.|\{|\}|\'|\"|\[|\]|\_|>|^|<|\t|=|#',docText)
    ret = []
    for e in tokenizedText:
        if e.lower() in stop_words:
            continue
        else:
            ret.append(e)
    return ret
def Sort_Tuple(tup):
    return (sorted(tup, key = lambda x: x[0], reverse=True))  


doc_struct_list = []
corpus=[]
corpus2 = []

i = 0

for subdir, dirs, files in os.walk(directory):
    for file in files:
        with open(subdir+'/'+file,mode="r", encoding = "utf8", errors="ignore") as f:
            doc_curr=f.read()
            corpus2.append(doc_curr)
            tokenizedText=tokenizerForDocs(doc_curr)
            stemmedText=stemmer(tokenizedText)
            i += 1
            file = file.split('.')[0]
            doc_struct_list.append(file)
            corpus.append(stemmedText)

bm25 = rank_bm25.BM25Okapi(corpus)

f=open('AILA_2019_dataset/Query_doc.txt','r')

query_file = f.read()

temp_queries = query_file.split("\n")

queries = []

for query in temp_queries:
	new_query = query.split("||")[1]
	queries.append(new_query)

out_dict = {}

count = 0

for query in queries:
    query=tokenizerForDocs(query)
    print(query)
    query=stemmer(query)
    arr = bm25.get_scores(query)
    temp = []
    for i in range(len(arr)):
        temp.append((arr[i], doc_struct_list[i]))
    top = Sort_Tuple(temp)
    out_dict.update({count : top})
    count += 1

with open('bm.pickle', 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)