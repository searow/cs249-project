import numpy as np
import gensim

words = ['mother', 'father', 'king', 'queen', 'actor', 'actress']#, 'apple', 'banana', 'pear', 'microsoft', 'google']

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
  import matplotlib.pyplot as plt
  from sklearn.decomposition import PCA

  print('Loading Model...')
  model = gensim.models.KeyedVectors.load_word2vec_format(
      'GoogleNews-vectors-negative300.bin', binary=True)

  print('Transforming Vectors...')
  pca = PCA(n_components=2)

  low_dim_embs = pca.fit_transform(model[words])
  # low_dim_embs = tsne.fit_transform(model[words])

  print('Plotting...')
  plot_with_labels(low_dim_embs, words, './pca_gensim.png')

except ImportError as ex:
  print(ex)