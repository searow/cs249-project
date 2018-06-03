from utils import *
import numpy as np
from scipy.stats import spearmanr
import gensim

window_size = 5
eval_fp = 'data/SCWS/ratings'

model = gensim.models.KeyedVectors.load_word2vec_format(
      'GoogleNews-vectors-negative300.bin', binary=True)
print('model loaded!')

# Load eval task.
eval_tests = load(eval_fp) # eval_test = list of dicts.
print('data loaded!')

def eval(eval_test, target_embeddings):
    d = {}
    for i in range(1, 3):
        word = eval_test['word{}'.format(i)]

        if(word in target_embeddings):
            d['w{}_real_meaning'.format(i)] = \
                target_embeddings[word]
        else:
            return False

    return cosine_sim(d['w1_real_meaning'], d['w2_real_meaning'])

def cosine_sim(emb1, emb2):
    return np.dot(emb1, emb2)

true_scores = []
test_scores = []
for eval_test in eval_tests:
    score = eval(eval_test, model)
    if score != False:
        test_scores.append(score)
        true_scores.append(eval_test['average_human_rating'])

spearman_score = spearmanr(test_scores, true_scores)
rho_correlation = spearman_score[0]
print('spearman rho: ', rho_correlation)
