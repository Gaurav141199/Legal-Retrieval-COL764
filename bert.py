import pandas as pd
import time
import sys
import os
import numpy as np
from translate import Translator
translator=Translator(to_lang='en',from_lang='es')
import sklearn.metrics.pairwise
from tqdm import tnrange
from sklearn.metrics import jaccard_score
import scipy
import re
import pickle
from nltk.corpus import stopwords
from PorterStemmer import PorterStemmer

stop_words = set(stopwords.words('english'))
stop_words.add(" ")
stop_words.add('\t')
# print(stop_words)

directory= r"AILA_2019_dataset/Object_casedocs/"

# def stemmer(tokenizedText):
#     ps=PorterStemmer()
#     stemmedText=[]
#     for word in tokenizedText:
#         stemmedText.append(ps.stem(word.lower(),0,len(word)-1))
#     return stemmedText
# def tokenizer(docText):
#     tokenizedText=re.split('/| |;|,|\*|\n|:|\(|\)|`|\.|\{|\}|\'|\"|\[|\]|\_|>|^|<|\t|=|#',docText)
#     tokenizedText=set(tokenizedText)
#     if '' in tokenizedText:
#         tokenizedText.remove('')
#     ret = tokenizedText.copy()
#     for e in tokenizedText:
#         if e.lower() in stopwords:
#             ret.remove(e)
#     return ret
# def tokenizerForDocs(docText):
#     tokenizedText=re.split('/| |;|,|\*|\n|:|\(|\)|`|\.|\{|\}|\'|\"|\[|\]|\_|>|^|<|\t|=|#',docText)
#     ret = []
#     for e in tokenizedText:
#         if e.lower() in stop_words:
#             continue
#         else:
#             ret.append(e)
#     return ret

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



def generate(stemmedText):
    ret = ""
    k = 0
    for i in range(len(stemmedText)):
        if i == 0:
            ret = ret + stemmedText[i]
        else:
            ret = ret + " " + stemmedText[i]
    return ret

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
            # ret = generate(stemmedText)
            i += 1
            file = file.split('.')[0]
            doc_struct_list.append(file)
            corpus.append(stemmedText)

# print(doc_struct_list)

raw_data = corpus

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('bert-base-nli-mean-tokens') #BERT BASE
# embedder = SentenceTransformer('bert-large-nli-stsb-mean-tokens') # LARGE BERT

corpus_embeddings=embedder.encode(raw_data)

f=open('AILA_2019_dataset/Query_doc.txt','r')

query_file = f.read()

temp_queries = query_file.split("\n")

queries = []
id_to_query = []

for query in temp_queries:
    new_query = query.split("||")[1]
    id_to_query.append(query.split("||")[0])
    queries.append(new_query)
    break

query_embeddings = embedder.encode(queries)

out_dict = {}

j = 0
# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 2914
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    # print("\n\n======================\n\n")
    # print("Query:", query)
    # print("\nTop 5 most similar sentences in corpus:")
    ret = []
    for idx, distance in results[0:closest_n]:
        if j == 0:
        	print(idx)
        # print(raw_data[idx], "(Score: %.4f)" % (1-distance))
        ret.append((doc_struct_list[idx], "(Score: %.4f)" % (1-distance)))

    out_dict.update({id_to_query[j] : ret})
    j += 1

with open('bert1_without_stop.pickle', 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(out_dict["AILA_Q1"][0:5])
