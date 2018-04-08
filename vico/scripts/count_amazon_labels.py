import argparse
import gzip
import pickle
from collections import Counter
from pprint import pprint
from .map_reduce import SimpleMapReduce
import multiprocessing
import os
from itertools import islice


def _parse_labels(path):
    print('Reading from', path)
    with open(path, 'r') as g:
        for l in g:
            yield eval(l)


def get_category(file):
    labels = list(_parse_labels(file))
    output = []
    for label in labels:
        if 'categories' in label:
            for category in label['categories']:
                output.append((tuple(category), 1))
        output.append((('No Categories',), 1))
    return output


def count_categories(mapped_value):
    category, occurrences = mapped_value
    return category, sum(occurrences)


def count(path):
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    print('performing map reduce')
    map_reduce = SimpleMapReduce(
        map_func=get_category,
        reduce_func=count_categories,
        num_workers=multiprocessing.cpu_count()
    )
    category_counts = map_reduce(files)
    category_counts = dict(category_counts)
    pprint(category_counts)
    with open('category_counts.pkl', 'wb') as f:
        pickle.dump(category_counts, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    count(**vars(args))
