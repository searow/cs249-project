import inspect, pickle
from collections import OrderedDict


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
    import numpy as np
    some_numpy_mat = np.arange(0, 6).reshape(2, 3)
    xxxxx = [1, 2, 3]
    y = 'some_str'
    c = {5: np.array([[7, 7, 7], [7, 7]])}
    save_as_dict('some_path_to_save', some_numpy_mat, some_numpy_mat, xxxxx, y, 1, True, c)
    print('Loaded\n', load_as_dict('some_path_to_save'))
    e = 5
    c = 2
    d = -1
    save_as_dict('x', e, 1000, b=c, f=d)

    print('Loaded\n{}'.format(load_as_dict('x.pickle')))


if __name__ == '__main__':
    main()
