from typing import Iterator

from sklearn.model_selection import KFold

from vico import train, read, args, report
from numpy import ndarray

from vico.pipes import preprocess, as_array
from pipe import take, as_tuple

import logging

log = logging.getLogger('vico.validate')


def _split(docs: ndarray) -> Iterator[ndarray]:
    splitter = KFold(n_splits=args.get().folds)
    return splitter.split(docs)


def k_cross():
    docs = read.all_docs() | take(5) | as_tuple
    pdocs = _preprocess(docs)
    folds = args.get().folds
    for fold, (train_indices, test_indices) in enumerate(_split(pdocs)):
        log.info('Starting fold %i of %i', fold + 1, folds)
        train_docs = pdocs[train_indices]
        network = train.early_stopping(train_docs)
        test_docs = pdocs[test_indices]
        report.save(network, train_docs, test_docs)


def _preprocess(docs):
    log.info('Pre-processing %i documents', len(docs))
    pdocs = (docs
             | preprocess.remove_useless_tags
             | preprocess.html_tokenize
             | as_array)
    return pdocs


if __name__ == '__main__':
    k_cross()



