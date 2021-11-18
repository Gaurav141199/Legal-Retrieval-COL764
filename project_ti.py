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


path = "AILA_2019_dataset/relevance_judgments_priorcases.txt"

f=open(path,'r')

temp_data1 = f.read()

temp_data = temp_data1.split("\n")

data = []

for e in temp_data:
    data.append(e.split())

def get_relevance(i, data):
    return data[i*2914: (i + 1)*2914]

query_relevance_map = {}

for i in range(50):
    query_relevance_map.update({i: get_relevance(i, data)})

print(query_relevance_map[0])

