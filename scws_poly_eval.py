from utils import *
import numpy as np
from scipy.stats import spearmanr

window_size = 5
embeddings_fp = 'saved_embeddings_step_5000000_k_3_dim_100_neg_65_swind_5_contexts_1_2018-05-31-23_37_08'
corpus_fp = 'data/word2vec_sample/text8_tokenized_50000'
eval_fp = 'data/SCWS/ratings'

# Load trained data.
train_loaded = load(embeddings_fp)
target_embeddings = train_loaded['target_embeddings']
target_counts = train_loaded['target_counts']
context_embeddings = train_loaded['context_embeddings']
context_counts = train_loaded['context_counts']

# Load corpus data.
corpus_loaded = load(corpus_fp)
dictionary = corpus_loaded['dictionary']
reversed_dictionary = corpus_loaded['reversed_dictionary']

# Load eval task.
eval_tests = load(eval_fp) # eval_test = list of dicts.

def eval(eval_test, dictionary,
         target_embeddings, target_mask, context_embeddings):
    d = {}
    for i in range(1, 3):
        word_index = eval_test['word{}_index'.format(i)]
        sentence = eval_test['word{}_in_context'.format(i)]
        tokenized_sentence = tokenize_sentence( \
            sentence, dictionary)
        all_meaning_embedding_list = extract_meanings_as_list(target_embeddings, \
            word_index, target_mask)
        context_embedding_list = extract_contexts_as_list(\
            tokenized_sentence, \
            word_index, context_embeddings)
        d['w{}_real_meaning'.format(i)] = get_real_meaning_embedding( \
            context_embedding_list, \
            all_meaning_embedding_list)

    return cosine_sim(d['w1_real_meaning'], d['w2_real_meaning'])

def tokenize_sentence(sentence, dictionary):
    # Chris
    # sentence = [0, 1, 5, 2, 1, 2, 0]

    unknown_token = 0
    words = sentence.split()
    tokenized = []
    for word in words:
        if word not in dictionary:
            tokenized.append(unknown_token)
        else:
            tokenized.append(   dictionary[word])
    return tokenized

def extract_meanings_as_list(target_embeddings, word_index, target_mask):
    k = len(target_embeddings)
    meanings = []

    for i in range(k):
        if word_index > len(target_mask):
            print(word_index)
        if(target_mask[1][word_index][i]):
            meanings.append(word_index)

    import pdb; pdb.set_trace()

    return meanings

def extract_contexts_as_list(tokenized_sentence, word_index, context_emb_mat):
    # Alex
    # e.g. tokenized_sentence=[0, 2, 1, 3, ...]
    # word_index = 1
    # context_emb_mat: V x D
    # Remeber to check bound when using window_size!

    if word_index < window_size:
        context_tokens = [token for token in tokenized_sentence[0:word_index+window_size+1] \
                    if token != tokenized_sentence[word_index]]
    else:
        context_tokens = [token for token in tokenized_sentence[word_index-window_size:word_index+window_size+1] \
                    if token != tokenized_sentence[word_index]]

    return [context_emb_mat[token] for token in context_tokens]

def get_real_meaning_embedding(context_embedding_list, all_meaning_embedding_list):
    # Jack
    avg_c = np.mean(context_embedding_list, axis=0)
    scores = [np.dot(m, avg_c) for m in all_meaning_embedding_list]
    return all_meaning_embedding_list[np.argmax(scores)]

def cosine_sim(emb1, emb2):
    return np.dot(emb1, emb2)

true_scores = []
test_scores = []
target_mask = filter_words(target_counts)
for eval_test in eval_tests:
    score = eval(eval_test, reversed_dictionary,
            target_embeddings, target_mask, context_embeddings)
    test_scores.append(score)
    true_scores.append(eval_test['average_human_rating'])

spearman_score = spearmanr(test_scores, true_scores)
rho_correlation = spearman_score[0]
print('spearman rho: ', rho_correlation)
