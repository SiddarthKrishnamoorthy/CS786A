import numpy as np
import gensim
import pickle as pkl

def loadWord2Vec(path):
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return model

model = loadWord2Vec("GoogleNews-vectors-negative300.bin")
def calculate_WV(words):
    word1, word2 = words
    return model.similarity(word1, word2)

with open('combined.csv', 'r') as f:
    data=f.read()
    data = data.split('\n')
    data.pop()
    data = data[1:]
    word1 = [x.split(',')[0] for x in data]
    word2 = [x.split(',')[1] for x in data]
    scores = [float(x.split(',')[2]) for x in data]
    words = list(zip(word1, word2))
    ngd_scores = list(map(calculate_WV, words))

results = {}
results['words'] = words
results['ngd'] = ngd_scores
with open('wv_score.pkl', 'wb') as f:
    pkl.dump(results, f)

