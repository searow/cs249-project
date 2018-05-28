import numpy as np
import pickle

k = 1
#fn = '/home/david/Documents/2018SPRING/249/cs249-project/output/saved_embeddings_orig_dim_128_2018-05-27-19:17:58/saved_embeddings_orig_step_100000_dim_128_2018-05-27-19:17:58.npy'
#fn = '/home/david/Documents/2018SPRING/249/cs249-project/output/saved_embeddings_k_1_dim_300_2018-05-27-17:15:50/saved_embeddings_step_100000_k_1_dim_300_2018-05-27-17:15:50.npy'
fn = 'output//saved_embeddings_k_3_dim_100_2018-05-27-17:22:49//saved_embeddings_step_100000_k_3_dim_100_2018-05-27-17:22:49.npy'
corpus = 'data//word2vec_sample//text8_tokenized_50000'
words = ['woman', 'man', 'king', 'queen']#'apple', 'banana', 'pear', 'microsoft', 'google'] 

def create_plotID_list(words, dictionary):
  return [dictionary[i] for i in words]

# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  print(low_dim_embs.shape)
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  print('Reading the tokenized corpus...')
  with open(corpus, 'rb') as readfile:
    read_obj = pickle.load(readfile)
    data = read_obj['data']
    count = read_obj['count']
    dictionary = read_obj['dictionary']
    reversed_dictionary = read_obj['reversed_dictionary']
    vocabulary_size = len(count)

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=50000, method='exact')
  plot_only = create_plotID_list(words, dictionary)
  data = np.load(fn)
  low_dim_embs_list = []
  labels_list = []

  for i in range(k):
    print(i)
    print(data[i].shape)
    low_dim_embs = tsne.fit_transform(data[i][plot_only, :])
    low_dim_embs_list.append(low_dim_embs)
    labels = [reversed_dictionary[j] + '_' + str(i) for j in plot_only]
    labels_list += labels

  plot_with_labels(np.concatenate(low_dim_embs_list), labels_list, './/tsne.png')

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)