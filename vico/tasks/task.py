import logging
from abc import ABC, abstractmethod
from typing import Tuple, Any

from keras import Model
from pandas import Series
from sklearn.model_selection import train_test_split

from f import List
from vico import preprocess
from vico.html_document import Tokenization
from vico.types import Tokenizations
from vico.vocabulary import Vocabulary
from numpy import ndarray


def split(tokenizations) -> Tuple[List[Tokenization], List[Tokenization]]:
    ts_series = Series(tokenizations)
    train_set, test_set = train_test_split(ts_series, test_size=.2)
    return List(train_set), List(test_set)


log = logging.getLogger('vico.task')


class Task(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def tokenizations(self) -> Tokenizations:
        return self._train_set + self._test_set

    @property
    def input_length(self):
        return preprocess.maxlen(self.tokenizations)

    def __init__(self,
                 train_tokenizations: Tokenizations,
                 test_tokenizations: Tokenizations,
                 vocabulary: Vocabulary,
                 shared_layers
                 ):
        self.unique_labels = set(
            self.label(t) for t in train_tokenizations + test_tokenizations
        )
        self._vocabulary = vocabulary
        self._shared_layers = shared_layers
        train_tokenizations = self.filter_tokenizations(train_tokenizations)
        self._train_set, self._test_set = split(train_tokenizations)
        self._model = self.compile_model()
        self.epoch = 0
        self.epochs_without_improvement = 0
        self.best_loss = float('inf')

    def get_train_batch(self) -> Tuple[ndarray, ndarray]:
        return self._vocabulary.make_batch(
            self._train_set,
            self.label
        )

    @abstractmethod
    def filter_tokenizations(self, tokenizations) -> Tokenizations:
        pass

    @abstractmethod
    def compile_model(self) -> Model:
        pass

    @abstractmethod
    def encode_labels(self, tokenization: Tokenizations) -> ndarray:
        pass

    @abstractmethod
    def label(self, tokenization: Tokenization) -> Any:
        pass

    def reset(self):
        self._model = self.compile_model()
        self.epoch = None
        self.epochs_without_improvement = None

    def fit_early_stopping(self):
        log.info('Fitting one epoch early stopping on task: %s', self.name)
        sequences, labels = self._vocabulary.make_batch(
            self._train_set,
            self.encode_labels
        )
        self._model.fit(
            sequences,
            labels,
            epochs=1
        )
        self.epoch += 1
        test_loss = self.test_loss()
        if test_loss < self.best_loss:
            log.info('New best loss found. '
                     'Resetting epochs without improvement for task: %s',
                     self.name)
            self.best_loss = test_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def fit(self):
        log.info('Fitting task %s on all data for 1 epoch', self.name)
        data = self._train_set + self._test_set
        sequences, labels = self._vocabulary.make_batch(data, self.encode_labels)
        self._model.fit(sequences, labels, epochs=1)

    def test_loss(self, tokenizations: Tokenizations = None):
        if tokenizations is None:
            sequences, labels = self._vocabulary.make_batch(
                self._test_set,
                self.encode_labels
            )
        else:
            tokenizations = self.filter_tokenizations(tokenizations)
            sequences, labels = self._vocabulary.make_batch(
                tokenizations,
                self.encode_labels
            )
        return self._model.evaluate(sequences, labels)
