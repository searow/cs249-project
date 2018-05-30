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

# Hardcoded Elements:
k = 3
folder_name = '/home/david/Documents/2018SPRING/249/cs249-project/output/saved_embeddings_k_3_dim_100_2018-05-28-20:52:23'

corpus = 'data/word2vec_sample/text8_tokenized_50000'
words = ['king', 'queen', 'actor', 'actress', 'father', 'mother']#'apple', 'banana', 'pear', 'microsoft', 'google'] 

prefix = 'pca_plotof_'

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

	plt.savefig(filename)
	plt.close()

try:
	set_plot_defaults()

	print('Reading the tokenized corpus...')
	with open(corpus, 'rb') as readfile:
		read_obj = pickle.load(readfile)
		data = read_obj['data']
		count = read_obj['count']
		dictionary = read_obj['dictionary']
		reversed_dictionary = read_obj['reversed_dictionary']
		vocabulary_size = len(count)

	pca = PCA(n_components=2)
	plot_only = create_plotID_list(words, dictionary)
	
	print('Plotting...')
	for bf in listdir(folder_name):
		# Check for existing plot files
		if bf.startswith(prefix):
			continue

		# Load vectors
		data = np.load(folder_name + '/' + bf)
		low_dim_embs_list = []
		labels_list = []

		# Condition embeddings/labels
		for i in range(k):
			low_dim_embs = pca.fit_transform(data[i][plot_only, :])
			low_dim_embs_list.append(low_dim_embs)
			if k == 1:
				labels = [reversed_dictionary[j] for j in plot_only]
			else:
				labels = [reversed_dictionary[j] + '_' + str(i) \
					for j in plot_only]
			labels_list += labels

		#Plot
		plot_with_labels(
			np.concatenate(low_dim_embs_list), labels_list, \
			folder_name + '/' + prefix + bf + '.png')

	print('Finished Plotting!')

except ImportError as ex:
	print(ex)