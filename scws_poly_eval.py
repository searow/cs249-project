from utils import *
import numpy as np
from scipy.stats import spearmanr

window_sizes = range(1, 100)
embeddings_fp = './embedding_results/saved_embeddings_step_5000000_k_2_dim_150_neg_1_swind_5_contexts_1_2018-06-02-20-39-00/saved_embeddings_step_5000000_k_2_dim_150_neg_1_swind_5_contexts_1_2018-06-02-20-39-00'
# embeddings_fp = 'saved_embeddings_step_5000000_k_1_dim_300_neg_65_swind_5_contexts_1_2018-06-01-13_41_24'
corpus_fp = './data/enwik9/enwik9_tokenized_50000'
eval_fp = 'data/SCWS/ratings'


def eval(eval_test, name_token,
         target_embeddings, target_mask, context_embeddings, window_size):
    d = {}
    for i in range(1, 3):
        word_index = eval_test['word{}_index'.format(i)]
        sentence = eval_test['word{}_in_context'.format(i)]
        tokenized_sentence = tokenize_sentence( \
            sentence, name_token)
        all_meaning_embedding_list = extract_meanings_as_list(target_embeddings, \
            tokenized_sentence, word_index, target_mask)
        context_embedding_list = extract_contexts_as_list(\
            tokenized_sentence, \
            word_index, context_embeddings, window_size)
        d['w{}_real_meaning'.format(i)] = get_real_meaning_embedding( \
            context_embedding_list, \
            all_meaning_embedding_list)

    return cosine_sim(d['w1_real_meaning'], d['w2_real_meaning'])

def tokenize_sentence(sentence, name_token):
    # Chris
    # sentence = [0, 1, 5, 2, 1, 2, 0]

    unknown_token = 0
    words = sentence.split()
    tokenized = []
    for word in words:
        if word not in name_token:
            tokenized.append(unknown_token)
        else:
            tokenized.append(name_token[word])
    return tokenized

def extract_meanings_as_list(target_embeddings, tokenized_sentence, word_index, target_mask):
    k = len(target_embeddings)
    meanings = []
    target_token = tokenized_sentence[word_index]
    is_untrained_embs = np.count_nonzero(target_mask[0][target_token]) == 0

    for i in range(k):
        if(is_untrained_embs or target_mask[1][target_token][i]):
            meanings.append(target_embeddings[i][target_token])

    return meanings

def extract_contexts_as_list(tokenized_sentence, word_index, context_emb_mat, window_size):
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

    assert(context_emb_mat.shape[0] == 1)
    return [context_emb_mat[0][token] for token in context_tokens]

def get_real_meaning_embedding(context_embedding_list, all_meaning_embedding_list):
    # Jack
    avg_c = np.mean(context_embedding_list, axis=0)
    scores = [np.dot(m, avg_c) for m in all_meaning_embedding_list]
    return all_meaning_embedding_list[np.argmax(scores)]

def cosine_sim(emb1, emb2):
    return np.dot(emb1, emb2)

def evaluate_spearman(window_size, eval_tests, name_token, context_embeddings,
                      target_counts, target_embeddings):
    true_scores = []
    test_scores = []
    word1_UNK = 0
    word2_UNK = 0
    eval_UNK = 0
    target_mask = filter_words(target_counts)
    for eval_test in eval_tests:
        if eval_test['word1'] not in name_token or eval_test['word2'] not in name_token:
            continue
        score = eval(eval_test, name_token,
                target_embeddings, target_mask, context_embeddings, window_size)
        test_scores.append(score)
        true_scores.append(eval_test['average_human_rating'])
    spearman_score = spearmanr(test_scores, true_scores)
    rho_correlation = spearman_score[0]
    return rho_correlation

if __name__ == '__main__':
    # Load trained data.
    train_loaded = load(embeddings_fp)
    target_embeddings = train_loaded['target_embeddings']
    target_counts = train_loaded['target_counts']
    context_embeddings = train_loaded['context_embeddings']
    context_counts = train_loaded['context_counts']

    # Load corpus data.
    corpus_loaded = load(corpus_fp)
    name_token = corpus_loaded['dictionary']
    token_name = corpus_loaded['reversed_dictionary']

    # Load eval task.
    eval_tests = load(eval_fp) # eval_test = list of dicts.

    spearman_scores = []
    for window in window_sizes:
        score = evaluate_spearman(window, eval_tests, name_token,
                                  context_embeddings,
                                  target_counts, target_embeddings)
        spearman_scores.append(score)
        print('{} {}'.format(window, score))
