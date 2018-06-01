import inspect, pickle, random
from collections import OrderedDict
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print('No matplotlib so good luck')
import numpy as np


def save_as_dict(filepath, *args, **kwargs):
    frames = inspect.getouterframes(inspect.currentframe())
    frame = frames[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    dict_to_save = OrderedDict()
    all_args_strs = string[string.find('(') + 1:-1].split(',')
    if 1 + len(args) + len(kwargs) != len(all_args_strs):
        msgs = ['Did you call this function in one line?', \
                'Did the arguments have comma "," in the middle?']
        raise RuntimeError('\n'.join(msgs))
    for i, name in enumerate(all_args_strs[1:]):
        if name.find('=') != -1:
            name = name.split('=')[1]
        name = name.strip()
        if i >= 0 and i < len(args):
            dict_to_save[name] = args[i]
        else:
            break
    dict_to_save.update(kwargs)
    print('Saving dictionary to {}:\n{}'.format(filepath, dict_to_save))
    save(filepath, dict_to_save)


def load_as_dict(filepath):
    return load(filepath)


def vis_train_count(filepath, title, counter_target, counter_context,
                    word_id, id_word, to_vis=['apple', 'amazon'],
                    num_to_vis=10):
    if type(filepath) is not str or type(title) is not str:
        raise RuntimeError('filepath and title must be valid strings')
    # if counter_target.shape != counter_context.shape:
    #     raise RuntimeError('counter_target and counter_context must be of the same shape')
    if type(word_id) is not dict or type(id_word) is not dict:
        raise RuntimeError('word_id and id_word must be valid dictionaries')
    if to_vis:
        ids = [word_id[word] for word in to_vis]
        apsb = '_{}'.format('_'.join(to_vis))
        aps = apsb + '_target'
        vis_train_count_helper(filepath + aps, title + aps, \
                               counter_target[ids], to_vis)
        aps = apsb + '_context'
        vis_train_count_helper(filepath + aps, title + aps, \
                               counter_context[ids], to_vis)
    if num_to_vis:
        if type(num_to_vis) is not int or num_to_vis <= 0:
            raise RuntimeError('num_to_vis must be a positive integer')
        V = counter_context.shape[0]
        ids = random.sample(range(0, V), num_to_vis)
        y_labels = [id_word[id] for id in ids]
        apsb = '_rand_{}'.format(num_to_vis)
        aps = apsb + '_target'
        vis_train_count_helper(filepath + aps, title + aps, \
                               counter_target[ids], y_labels)
        aps = apsb + '_context'
        vis_train_count_helper(filepath + aps, title + aps, \
                               counter_context[ids], y_labels)


def vis_train_count_helper(filepath, title, mat, y_labels):
    fig, ax = plt.subplots()
    im = ax.imshow(mat)
    ax.figure.colorbar(im, ax=ax)
    k = mat.shape[1]
    for i in range(len(y_labels)):
        for j in range(k):
            ax.text(j, i, mat[i, j], ha="center", va="center", color="w")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title)
    print('Saving plot to {}'.format(filepath))
    plt.savefig(filepath)


def save(filepath, obj):
    with open(proc_filepath(filepath), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(filepath):
    with open(proc_filepath(filepath), 'rb') as handle:
        return pickle.load(handle)


def proc_filepath(filepath):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    ext = '.pickle'
    if ext not in filepath:
        filepath += ext
    return filepath


def main():
    # some_numpy_mat = np.arange(0, 6).reshape(2, 3)
    # xxxxx = [1, 2, 3]
    # y = 'some_str'
    # c = {5: np.array([[7, 7, 7], [7, 7]])}
    # save_as_dict('some_path_to_save',some_numpy_mat, some_numpy_mat, xxxxx, y, 1, True, c)
    # print('Loaded\n',load_as_dict('some_path_to_save'))
    # e = 5
    # c = 2
    # d = -1
    # save_as_dict('x', e, 1000, b=c, f=d)
    # print('Loaded\n{}'.format(load_as_dict('x.pickle')))

    import string
    V = 1000
    k = 3
    counter_target = np.arange(0, V * k).reshape(V, k)
    counter_context = np.random.randint(1000, size=(V, k))
    name_token = {}
    token_name = {}
    for i in range(V):
        word = ''.join(random.choice(string.ascii_lowercase) \
                       for _ in range(random.randint(1, 20)))
        if i == 0:
            word = 'apple'
        if i == 1:
            word = 'amazon'
        name_token[word] = i
        token_name[i] = word
    vis_train_count('some_picture', 'this is a plot', counter_target, \
                    counter_context, name_token, token_name)


if __name__ == '__main__':
    main()
