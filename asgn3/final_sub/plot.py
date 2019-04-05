import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def plot(path, is_ngd = True):
    """
    args:
        path: Path to pickle file containing word pairs and similarity scores
        is_ngd: If plot of ngd, set it to True, if plot of word2vec, set it to False

    returns:
        None
    """
    with open(path, 'rb') as f:
        data = pkl.load(f)

    words = data['words']
    ngd_scores = data['ngd']
    with open('combined.csv', 'r') as f:
        data=f.read()
        data = data.split('\n')
        data.pop()
        data = data[1:]
        word1 = [x.split(',')[0] for x in data]
        word2 = [x.split(',')[1] for x in data]
        scores = [float(x.split(',')[2]) for x in data]

    scores = np.asarray(scores)
    ngd_scores = np.asarray(ngd_scores)
    
    ngd_scores = np.absolute(ngd_scores)
    ngd_scores = ngd_scores - np.min(ngd_scores)
    ngd_scores = ngd_scores/np.max(ngd_scores)*10.0
    if is_ngd:
        ngd_scores = 10.0 - ngd_scores
    
    plt.scatter(scores, ngd_scores, facecolors='none', edgecolors='b')
    plt.xlabel('Human similarity ratings')
    if is_ngd:
        plt.ylabel('Scaled NGD')
    else:
        plt.ylabel('Scaled Word2Vec')
    plt.savefig('plot.png')

plot('wv_score.pkl', False)
