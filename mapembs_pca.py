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
from utils import load_as_dict
import pandas as pd

# Hardcoded Elements:
k = 3
folder_name = '/home/david/Documents/2018SPRING/249/cs249-project/output/models_k_3_dim_100_neg_10_swind_5_2018-05-31-15:53:01'

corpus = 'data/word2vec_sample/text8_tokenized_50000'
words = ['king', 'queen', 'actor', 'actress', 'father', 'mother']#'apple', 'banana', 'pear', 'microsoft', 'google'] 

prefix_tgt = 'pca_plotof_tgt'
prefix_nce = 'pca_plotof_ctx'

# Make sure that the corpus file exists before continuing.
if not path.isdir(folder_name):
		print('The output folder could not be found')
		exit(1)

def create_plotID_list(words, dictionary):
	return [dictionary[i] for i in words]

def set_plot_defaults():
	font = {'size'   : 25}
	matplotlib.rc('font', **font)

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

		axes = plt.gca()
		axes.set_xlim([-1,1])
		axes.set_ylim([-1,1])

	plt.savefig(filename)
	plt.close()

try:
	set_plot_defaults()

	print('Reading the tokenized corpus...')
	read_obj = load_as_dict(corpus)
	dictionary = read_obj['dictionary']
	reversed_dictionary = read_obj['reversed_dictionary']

	pca = PCA(n_components=2)
	plot_only = create_plotID_list(words, dictionary)
	
	print('Plotting...')
	for bf in listdir(folder_name):
		# Check for existing plot files
		if bf.startswith(prefix_tgt) or bf.startswith(prefix_nce):
			continue

		# Load vectors
		loaded_data = load_as_dict(folder_name + '/' + bf)
		tgt_embs = loaded_data['target_embeddings']
		tgt_cnts = loaded_data['target_counts']
		nce_embs = loaded_data['context_embeddings']
		nce_cnts = loaded_data['context_counts']

		low_dim_embs_list = []
		labels_list = []

		# Condition embeddings/labels
		for i in range(k):
			low_dim_embs = tgt_embs[i][plot_only, :]
			low_dim_embs_list.append(low_dim_embs)
			if k == 1:
				labels = [reversed_dictionary[j] for j in plot_only]
			else:
				labels = [reversed_dictionary[j] + '_' + str(i) \
					+ '_{}'.format(int(tgt_cnts[j][i])) \
					for j in plot_only]
			labels_list += labels

		# import pdb; pdb.set_trace()
		low_dim_embs_list = pca.fit_transform(np.concatenate(low_dim_embs_list))

		#Plot
		plot_with_labels(
			low_dim_embs_list, labels_list, \
			folder_name + '/' + prefix_tgt + bf + '.png')

		low_dim_embs_list = []
		labels_list = []

		# Condition embeddings/labels
		for i in range(k):
			low_dim_embs = nce_embs[i][plot_only, :]
			low_dim_embs_list.append(low_dim_embs)
			if k == 1:
				labels = [reversed_dictionary[j] for j in plot_only]
			else:
				labels = [reversed_dictionary[j] + '_' + str(i) \
					+ '_{}'.format(int(nce_cnts[j][i])) \
					for j in plot_only]
			labels_list += labels

		low_dim_embs_list = pca.fit_transform(np.concatenate(low_dim_embs_list))

		#Plot
		plot_with_labels(
			low_dim_embs_list, labels_list, \
			folder_name + '/' + prefix_nce + bf + '.png')

	print('Finished Plotting!')

except ImportError as ex:
	print(ex)