#!/usr/bin/env python

import urllib.request
import time
import math
import json
from bs4 import BeautifulSoup
import re
import numpy as np
from copy import deepcopy
import pickle as pkl

def query_google(word):
    user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
    headers = { 'User-Agent' : user_agent }
    query = 'https://www.google.com/search?client=firefox-b-d&q=' + word
    req = urllib.request.Request(query, headers = headers)
    fil = urllib.request.urlopen(req)
    f = fil.read().decode('utf-8')
    count = None
    if "About" in f:
        l = re.findall(r'About ([0-9]+(,[0-9]+)+) results', f)
        count = int(l[0][0].replace(',',''))
    return count

def calculate_NGD(words):
    word1, word2 = words
    len_N = 30000000000000
    len_word1 = query_google(word1)
    len_word2 = query_google(word2)
    len_word1_word2 = query_google(word1+'+'+word2)
    ngd = (max(math.log(len_word1),math.log(len_word2)) - math.log(len_word1_word2))/(math.log(len_N) - min(math.log(len_word1),math.log(len_word2)))
    return ngd

ngd_scores = []
with open('combined.csv', 'r') as f:
    data=f.read()
    data = data.split('\n')
    data.pop()
    data = data[1:]
    word1 = [x.split(',')[0] for x in data]
    word2 = [x.split(',')[1] for x in data]
    scores = [float(x.split(',')[2]) for x in data]
    
    words = list(zip(word1, word2))
    
    ctr = 0
    for word in words:
        print("Word pair {0}".format(ctr+1))
        score = calculate_NGD(word)
        time.sleep(2)
        print(score)
        ngd_scores.append(deepcopy(score))
        ctr+=1
    #ngd_scores = list(map(calculate_NGD, words))

result = {}
result['ngd'] = ngd_scores
result['words'] = words
with open('ngd_scores.pkl', 'wb') as f:
    pkl.dump(result, f)
