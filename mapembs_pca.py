# Usage:
#
# folder_name = output folder with the saved embeddings
# k = number of embeddings per word
# corpus = tokenized corpus file
# words = words to plot
#
# If testing word embeddings from word2vec_basic.py:
#	k = 1
#	And change:
#		pca.fit_transform(data[i][plot_only, :])
#		pca.fit_transform(data[plot_only, :])

from os import path, listdir
from sys import exit
import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from utils import load_as_dict, filter_words

# Hardcoded Elements:
tgt_k = 1
nce_k = 1
folder_name = '/home/david/Documents/2018SPRING/249/cs249-project/output/models_k_1_dim_300_neg_65_swind_5_2018-06-01-13:41:24'

corpus = 'data/word2vec_sample/text8_tokenized_50000'
words = ['china', 'beijing', 'japan', 'tokyo', 'spain', 'madrid', 'germany', 'berlin', 'turkey', 'ankara']
#'migrant', 'reminded', 'obtaining', 'confused', 'subdivision', 'reich']#'apple', 'banana', 'pear', 'microsoft', 'google'] 
#'king', 'queen', 'actor', 'actress', 'father', 'mother',

prefix = 'pca_plot_'
prefix_lcl_tgt = prefix + 'lclfit_tgt_'
prefix_lcl_nce = prefix + 'lclfit_ctx_'
prefix_glb_tgt = prefix + 'glbfit_tgt_'
prefix_glb_nce = prefix + 'glbfit_ctx_'

# Make sure that the corpus file exists before continuing.
if not path.isdir(folder_name):
		print('The output folder could not be found')
		exit(1)

def create_plotID_list(words, dictionary):
	return [dictionary[i] for i in words]

def set_plot_defaults():
	font = {'size'   : 25}
	matplotlib.rc('font', **font)

def get_embs_labels(k, allembs, allcnts, reverse_dict, plot_only):
	embs_list = []
	labels_list = []
	norm_cnts, mask = filter_words(allcnts, 0.25)
	is_untrained_embs = np.count_nonzero(norm_cnts == 0)

	for i in range(k):
		embs = []
		labels = []

		for j in plot_only:
			if(mask[j][i] or is_untrained_embs):
				embs.append(allembs[i][j, :])

		embs_list.append(embs)

		if k == 1:
			for j in plot_only:
				if(mask[j][i] or is_untrained_embs):
					labels.append(
						reverse_dict[j]+'_{}'.format(int(allcnts[j][i])))
		else:
			for j in plot_only:
				if(mask[j][i] or is_untrained_embs):
					labels.append(reverse_dict[j]+'_'+str(i) \
						+ '_{}'.format(int(allcnts[j][i])))

		labels_list += labels

	return embs_list, labels_list

def plot_with_built_labels(low_dim_embs, labels, folder_name, \
							prefix, embFileN):
	plot_with_labels(low_dim_embs, labels, folder_name + '/' + prefix + embFileN + '.png')

# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
	assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
	plt.figure(figsize=(18, 18))  # in inches
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i, :]
		plt.scatter(x, y, s=400)
		plt.annotate(
				label,
				xy=(x, y),
				xytext=(5, 2),
				textcoords='offset points',
				ha='right',
				va='bottom',
				fontsize=35)

		# axes = plt.gca()
		# axes.set_xlim([-1,1])
		# axes.set_ylim([-1,1])

	plt.savefig(filename)
	plt.close()

try:
	set_plot_defaults()

	print('Reading the tokenized corpus...')
	read_obj = load_as_dict(corpus)
	dictionary = read_obj['dictionary']
	reversed_dictionary = read_obj['reversed_dictionary']

	pca_lcl_tgt = PCA(n_components=2)
	pca_lcl_nce = PCA(n_components=2)
	pca_glb_tgt = PCA(n_components=2)
	pca_glb_nce = PCA(n_components=2)
	plot_only = create_plotID_list(words, dictionary)
	
	print('Plotting...')
	for bf in listdir(folder_name):
		# Check for existing plot files
		if bf.startswith(prefix):
			continue

		print('Plotting {}'.format(bf))

		# Load vectors
		loaded_data = load_as_dict(folder_name + '/' + bf)
		tgt_embs = loaded_data['target_embeddings']
		tgt_cnts = loaded_data['target_counts']
		nce_embs = loaded_data['context_embeddings']
		nce_cnts = loaded_data['context_counts']

		# Fit the pca to all embeddings
		pca_glb_tgt.fit(np.concatenate(tgt_embs))
		pca_glb_nce.fit(np.concatenate(nce_embs))

		# Plot target Words
		high_dim_embs_list, labels_list = get_embs_labels(
				tgt_k, tgt_embs, tgt_cnts, reversed_dictionary, plot_only)

		pca_lcl_tgt.fit(np.concatenate(high_dim_embs_list))

		low_dim_embs_list = pca_lcl_tgt.transform(
				np.concatenate(high_dim_embs_list))
		plot_with_built_labels(
				low_dim_embs_list, labels_list, folder_name, prefix_lcl_tgt, bf)

		low_dim_embs_list = pca_glb_tgt.transform(
				np.concatenate(high_dim_embs_list))
		plot_with_built_labels(
				low_dim_embs_list, labels_list, folder_name, prefix_glb_tgt, bf)

		# Plot context Words
		high_dim_embs_list, labels_list = get_embs_labels(
				nce_k, nce_embs, nce_cnts, reversed_dictionary, plot_only)
		
		pca_lcl_nce.fit(np.concatenate(high_dim_embs_list))

		low_dim_embs_list = pca_lcl_nce.transform(
				np.concatenate(high_dim_embs_list))
		plot_with_built_labels(
				low_dim_embs_list, labels_list, folder_name, prefix_lcl_nce, bf)

		low_dim_embs_list = pca_glb_nce.transform(
				np.concatenate(high_dim_embs_list))
		plot_with_built_labels(
				low_dim_embs_list, labels_list, folder_name, prefix_glb_nce, bf)

	print('Finished Plotting!')

except ImportError as ex:
	print(ex)