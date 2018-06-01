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

def eval(eval_test, dictionary, reversed_dictionary,
         target_embeddings, target_counts, context_embeddings,
         counter_mask):
    d = {}
    for i in range(1, 3):
        word_index = eval_test['word{}'.format(i)]
        sentence = eval_test['word{}_in_context'.format(i)]
        tokenized_sentence = tokenize_sentence( \
            sentence, dictionary)
        all_meaning_embedding_list = extract_meanings_as_list(target_embeddings, \
            word_index, counter_mask)
        context_embedding_list = extract_contexts_as_list(\
            tokenized_sentence, \
            word_index, context_embeddings)
        d['w{}_real_meaning'.format(i)] = get_real_meaning_embedding( \
            context_embedding_list, \
            all_meaning_embedding_list)

    return cosine_sim(d['w1_real_meaning'], d['w2_real_meaning'])



    # Word2vec
    # a_emb = nemb[a]  # a's embs
    # b_emb = nemb[b]  # b's embs


    # # We expect that d's embedding vectors on the unit hyper-sphere is
    # # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    # target = c_emb + (b_emb - a_emb)

    # # Compute cosine distance between each pair of target and vocab.
    # # dist has shape [N, vocab_size].
    # dist = tf.matmul(target, nemb, transpose_b=True)

    # # For each question (row in dist), find the top 4 words.
    # _, pred_idx = tf.nn.top_k(dist, 4)

    # # Nodes for computing neighbors for a given word according to
    # # their cosine distance.
    # nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    # nearby_emb = tf.gather(nemb, nearby_word)
    # nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    # nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
    #                                      min(1000, self._options.vocab_size))

    # # Nodes in the construct graph which are used by training and
    # # evaluation to run/feed/fetch.
    # self._analogy_a = analogy_a
    # self._analogy_b = analogy_b
    # self._analogy_c = analogy_c
    # self._analogy_pred_idx = pred_idx
    # self._nearby_word = nearby_word
    # self._nearby_val = nearby_val
    # self._nearby_idx = nearby_idx

    return np.random.randint(0, 100)

def tokenize_sentence(sentence, dictionary):
    return [0, 1, 2]

def extract_meanings_as_list(target_embeddings, word_index, counter_mask):
    return []

def extract_contexts_as_list(tokenized_sentence, word_index, context_emb_mat):
    return []

def get_real_meaning_embedding(context_embedding_list, all_meaning_embedding_list):
    return []

def cosine_sim(emb1, emb2):
    return np.dot(emb1, emb2)

true_scores = []
test_scores = []
for eval_test in eval_tests:
    score = eval(eval_test, dictionary, reversed_dictionary,
            target_embeddings, target_counts, context_embeddings,
            context_counts)
    test_scores.append(score)
    true_scores.append(eval_test['average_human_rating'])

spearman_score = spearmanr(test_scores, true_scores)
rho_correlation = spearman_score[0]
print('spearman rho: ', rho_correlation)

