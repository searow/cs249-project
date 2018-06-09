from utils import *

match_list = ['apple', 'microsoft', 'google', 'banana', 'pear']
embedding_filepath = 'projector-test/saved_embeddings_step_17000000_k_2_dim_150_neg_10_swind_5_contexts_1_2018-06-06-23-20-25'
corpus_filepath = 'data/enwik9/enwik9_tokenized_50000'
output_embeddings_filepath = 'projector_embeddings.tsv'
output_metadata_filepath = 'projector_metadata.tsv'

print('Loading data')
embedding_data_load = load_as_dict(embedding_filepath)
corpus_data_load = load_as_dict(corpus_filepath)

targets = embedding_data_load['target_embeddings']
token_to_name = corpus_data_load['reversed_dictionary']

print('Writing to file')
with open(output_embeddings_filepath, 'w') as embed_file:
    with open(output_metadata_filepath, 'w') as meta_file:
        for k_val in range(targets.shape[0]):
            for vocab_idx in range(targets.shape[1]):
                if token_to_name[vocab_idx] in match_list:
                    for embed_idx in range(targets.shape[2]):
                        embed_file.write('{}\t'.format(targets[k_val, vocab_idx, embed_idx]))
                    embed_file.write('\n')
                    meta_file.write('{}_{}\n'.format(token_to_name[vocab_idx], k_val))
