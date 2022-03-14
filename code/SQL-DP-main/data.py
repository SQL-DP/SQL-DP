from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import open, super, object
import six
import random
import pickle
import itertools
import logging


logger = logging.getLogger(__name__)


class Node(object):
    def __init__(self):
        self.parent = None
        self.children = []
        self.type = ''


#  choose python2 or python3
if six.PY2:
    class NodeUnpickler(pickle.Unpickler, object):
        def find_class(self, module, name):
            if module == '__main__' and name == 'Node':
                return Node
            else:
                return super().find_class(module, name)
else:
    class NodeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__' and name == 'Node':
                return Node
            else:
                return super().find_class(module, name)


# function load_dataset(basename) invoke
class DataSet(object):
    def __init__(self, word2int):
        self.word2int = word2int
        self.splits = {}

    def _add_split(self, name, length, gen):
        self.splits[name] = (length, gen)

    def get_split(self, split):
        """split = 'train' or 'val' or 'test'
        returns generator
        """
        if split == 'all':
            total_len = sum([l for l, _ in self.splits.values()])
            return total_len, itertools.chain(*[gen for _, gen in self.splits.values()])
        return self.splits[split]


def _get_filename(basename, split):
    return '{}.{}.obj'.format(basename, split)


def load_dataset(basename):
    # function _get_filename()
    with open(_get_filename(basename, 'meta'), 'rb') as f:
        word2int = pickle.load(f)  # type is dict
        splits = pickle.load(f)  # type is dict

    def loader(f):
        try:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
        finally:
            f.close()

    # class Dataset(object)
    ds = DataSet(word2int)

    for split in splits:
        f = open(_get_filename(basename, split), 'rb')
        try:
            length = pickle.load(f)
        except:
            f.close()
            raise
        gen = loader(f)
        ds._add_split(split, length, gen)
    return ds
