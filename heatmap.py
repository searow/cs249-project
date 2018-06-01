from utils import *

data_fp = 'output/models_k_3_dim_100_neg_10_swind_5_2018-05-31-18-28-36/saved_embeddings_step_70000_k_3_dim_100_neg_10_swind_5_contexts_1_2018-05-31-18-28-36'
corpus_fp = 'data/word2vec_sample/text8_tokenized_50000'
output_filename = 'heatmap_70k_iterations_1_context'

data_loaded = load_as_dict(data_fp)
corpus_loaded = load_as_dict(corpus_fp)

plot_title = '70k_iterations_1_context'
context_counts = data_loaded['context_counts']
target_counts = data_loaded['target_counts']
name_token = corpus_loaded['dictionary']
token_name = corpus_loaded['reversed_dictionary']
visualized_words = ['apple', 'amazon', 'king', 'queen', 'chris', 'bat', 'bank', 'man', 'woman']
number_to_visualize = 20

vis_train_count(output_filename, plot_title, target_counts,
                context_counts, name_token, token_name, to_vis=visualized_words,
                num_to_vis=number_to_visualize)
