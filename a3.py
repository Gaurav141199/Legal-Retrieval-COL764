import networkx as nx
import os
import pandas as pd
import numpy as np
import json
import sys
import re
from PorterStemmer import PorterStemmer as ps
from bs4 import BeautifulSoup as bs
import math
import pickle
from nltk.corpus import stopwords

stop_words1 = set(stopwords.words('english'))

stop_words = set([])

for e in stop_words1:
	l = re.split(' |;|,|\*|\n|:|\(|\)|`|\.|\{|\}|\'|\"|\[|\]|\_|>|^|<|\t|=|#',e)
	for e1 in l:
		stop_words.add(e1)

print('on' in stop_words)

method = "cosine"

path = "AILA_2019_dataset/Object_casedocs/"

dir_list = os.listdir(path)

f=open('AILA_2019_dataset/Query_doc.txt','r')

query_file = f.read()

temp_queries = query_file.split("\n")

queries = []

for query in temp_queries:
	new_query = query.split("||")[1]
	queries.append(new_query)

def stem_words(list_words):
	words_set = set([])
	for e in list_words:
		if e == "":
			continue
		elif e.lower() in stop_words:
			continue
		else:
			port = ps()
			words_set.add(port.stem(e.lower(), 0 , len(e) - 1))
	return words_set

def checkKey(inverted_index, key):      
    if key in inverted_index:
        return True
    else:
        return False

def stem_words1(list_words):
	word_map = {}
	for e in list_words:
		if e == "":
			continue
		elif e.lower() in stop_words:
			continue
		else:		
			port = ps()
			word = port.stem(e.lower(), 0 , len(e) - 1)
			if checkKey(word_map, word):
				word_map[word] += 1
			else:
				word_map1 = {word: 1}
				word_map.update(word_map1)
	return word_map


def Sort_Tuple(tup):
    return (sorted(tup, key = lambda x: x[1], reverse=True))  

def get_matrix(n):
	matrix = []
	for i in range(n):
		add = []
		for j in range(n):
			add.append(0)
		matrix.append(add)
	return matrix

def get_score(j, k, id_to_doc_set):
	set1 = id_to_doc_set[j]
	set2 = id_to_doc_set[k]
	set3 = set1.intersection(set2)
	set4 = set1.union(set2)
	return len(set3)/len(set4)

def get_score_1(j, k, l):
	arr1 = l[j]
	arr2 = l[k]
	val1 = np.dot(arr1, np.transpose(arr2))
	val2 = np.sqrt(np.sum(np.square(arr1)))
	val3 = np.sqrt(np.sum(np.square(arr2)))
	return val1/(val2*val3)

def get_score_2(arr1, arr2):
	val1 = np.dot(arr1, np.transpose(arr2))
	val2 = np.sqrt(np.sum(np.square(arr1)))
	val3 = np.sqrt(np.sum(np.square(arr2)))
	return val1/(val2*val3)

def get_all_vec(idf_count, n, id_to_doc_map, word_to_num):
	l = []
	for i in range(len(id_to_doc_map)):
		map1 = id_to_doc_map[i]
		list1 = [0]*n
		for key in map1.keys():
			list1[word_to_num[key]] = math.log2(1 + (len(id_to_doc_map)/idf_count[key]))*math.log2(1 + map1[key])
		arr1 = np.array(list1)
		l.append(arr1)
	return l

def generate_qvec(words_map, idf_count, word_to_num, n, id_to_doc_map, vocabulary):
	list1 = [0]*n
	for key in words_map.keys():
		if key not in vocabulary:
			continue
		list1[word_to_num[key]] = math.log2(1 + (len(id_to_doc_map)/idf_count[key]))*math.log2(1 + words_map[key])
	arr1 = np.array(list1)
	return arr1

if method == "cosine":
	doc_to_id = {}
	id_to_doc = []
	id_to_doc_map = [] 
	vocabulary = set([])
	idf_count = {}
	i = 0
	for subdir, dirs, files in os.walk(path):
		for file in files:
			with open(subdir+'/'+file,mode="r", encoding = "utf8", errors="ignore") as f:
				curr_id = i
				data = f.read()
				list_words = re.split(' |;|,|\*|\n|:|\(|\)|`|\.|\{|\}|\'|\"|\[|\]|\_|>|^|<|\t|=|#',data)
				words_map = stem_words1(list_words)
				words_set = stem_words(list_words)
				# print(words_set)
				# add_set = set([])
				for word in words_set:
					if word == '':
						continue
					else:					
						if word in vocabulary:
							idf_count[word] += 1 
						else:
							idf_count1 = {word: 1}
							idf_count.update(idf_count1)
				vocabulary = vocabulary.union(words_set)
				doc_to_id.update({file.split(".")[0]: i})
				id_to_doc.append(file.split(".")[0])
				id_to_doc_map.append(words_map)
				i += 1
	n = len(vocabulary)
	word_to_num = {}
	num_to_word = {}
	count = 0
	for word in vocabulary:
		word_to_num.update({word : count})
		num_to_word.update({count : word})
		count += 1
	l = get_all_vec(idf_count, n, id_to_doc_map, word_to_num)
	final_score = {}
	count = 0
	for query in queries:
		list_words = re.split(' |;|,|\*|\n|:|\(|\)|`|\.|\{|\}|\'|\"|\[|\]|\_|>|^|<|\t|=|#',query)
		words_map = stem_words1(list_words)
		words_set = stem_words(list_words)
		query_vector = generate_qvec(words_map, idf_count, word_to_num, n, id_to_doc_map, vocabulary)
		score_to_doc = []
		for i in range(len(id_to_doc)):
			val = get_score_2(query_vector, l[i])
			score_to_doc.append((id_to_doc[i], val))
		ranked_list = Sort_Tuple(score_to_doc)
		final_score.update({count: ranked_list})	
		count += 1	
	with open('tfidf.pickle', 'wb') as handle:
		pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print(len(final_score))