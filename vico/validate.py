from f import Reader, Unary
from pandas import Series
from sklearn.model_selection import LeaveOneGroupOut
from itertools import islice

from vico import train, read, report, args

from vico.config import Config
from vico import preprocess
from vico.vocabulary import Vocabulary

import logging

from vico.types import Folds, Tokenizations, DocIterator
from . import configure_root_logger

log = logging.getLogger('vico.validate')


def split(tokenizations: Tokenizations) -> Reader[Config, Folds]:
    def _(config: Config) -> Reader[Config, Folds]:
        def generate() -> Folds:
            ts_series = Series(tokenizations)
            groups = [t.document.language for t in tokenizations]
            labels = [t.document.brand for t in tokenizations]
            folds = LeaveOneGroupOut().split(ts_series, labels, groups)
            for train_indices, test_indices in folds:
                train_set = ts_series[train_indices]
                test_set = ts_series[test_indices]
                yield (Tokenizations(train_set),
                       Tokenizations(test_set))
        return Reader.pure(generate())
    return Reader.ask(Config) >> _


def validate(folds: Folds) -> Reader[Config, None]:
    def _(config: Config):
        for fold, (train_docs, test_docs) in enumerate(folds):
            log.info('Starting fold %i of %i', fold + 1, config.folds)
            vocabulary = Vocabulary(train_docs, test_docs)
            fitted_tasks = train.early_stopping(
                train_docs,
                test_docs,
                vocabulary,
                config
            )
            report.save(fitted_tasks, train_docs, test_docs, config)
        return Reader.pure(None)
    return Reader.ask(Config) >> _


def limit(n: int) -> Unary[DocIterator, Reader[Config, DocIterator]]:
    def _(docs: DocIterator) -> Reader[Config, DocIterator]:
        return Reader.pure(docs[:n])
    return _


def leave_one_language_out() -> Reader[Config, None]:
    return (read.from_csv() >>
            limit(100) >>
            split >>
            validate)


if __name__ == '__main__':
    def run() -> None:
        config = args.get()
        configure_root_logger(config)
        leave_one_language_out()(config)
    run()
