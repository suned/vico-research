from pymonad import List
from pandas import Series
from sklearn.model_selection import KFold
from itertools import islice

from vico import train, read, report, args

from vico.config import Config
from vico import preprocess

import logging

from vico.types import Folds, DocIterator, Docs
from . import configure_root_logger

log = logging.getLogger('vico.validate')


def _split(docs: Docs, config: Config) -> Folds:
    doc_series = Series(docs)
    splitter = KFold(n_splits=config.folds)
    folds = splitter.split(doc_series)
    for train_indices, test_indices in folds:
        train_docs = doc_series[train_indices]
        test_docs = doc_series[test_indices]
        yield Docs(*train_docs), Docs(*test_docs)


def k_cross(config: Config) -> None:
    docs = read.all_docs(config)
    docs = islice(docs, 5)
    pdocs = _preprocess(config, docs)
    folds = config.folds
    for fold, (train_docs, test_docs) in enumerate(_split(pdocs, config)):
        log.info('Starting fold %i of %i', fold + 1, folds)
        network = train.early_stopping(train_docs, config)
        report.save(network, train_docs, test_docs, config)


def _preprocess(config: Config, docs: DocIterator) -> Docs:
    ds = List(*docs)
    log.info('Pre-processing %i documents', len(ds))
    tokenize = preprocess.html_tokenize(config.use_attributes)
    pipeline = (preprocess.lowercase *
                tokenize *
                preprocess.remove_useless_tags)

    return pipeline(ds)


if __name__ == '__main__':
    def run() -> None:
        config = args.get()
        configure_root_logger(config)
        k_cross(config)
    run()



