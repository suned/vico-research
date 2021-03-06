import logging
from abc import ABC, abstractmethod
from typing import Tuple, Any, List

from keras import Model
from keras.models import load_model
from pandas import Series
from serum import inject, immutable
from sklearn.model_selection import train_test_split

from vico import preprocess
from vico.cross_validation_split import CrossValidationSplit
from vico.html_document import HTMLDocument
from vico.shared_layers import SharedLayers
from vico.vocabulary import Vocabulary
from vico.console_arguments import ConsoleArguments
from numpy import ndarray
import os


def split(documents) -> Tuple[List[HTMLDocument], List[HTMLDocument]]:
    doc_series = Series(documents)
    train_set, test_set = train_test_split(doc_series, test_size=.2)
    return list(train_set), list(test_set)


log = logging.getLogger('vico.task')


class Task(ABC):
    args = inject(ConsoleArguments)
    cross_validation_split = inject(CrossValidationSplit)
    target = immutable(False)
    vocabulary = inject(Vocabulary)

    def save(self):
        folder = self.args.get().model_folder
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, self.name + '.hd5')
        log.info('Saving model: %s to %s', self.name, path)
        self._model.save(path)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def scoring_function(self) -> str:
        pass

    @property
    def input_length(self) -> int:
        return preprocess.maxlen(self.cross_validation_split.documents)

    def __init__(self, shared_layers: SharedLayers, all_data=False):
        self._shared_layers = shared_layers
        labels = [self.label(d) for d in self.filter_documents(
                self.cross_validation_split.documents
            )]
        if labels and type(labels[0]) != list:
            self.unique_labels = set(l for l in labels)
        else:
            self.unique_labels = set(l for ls in labels for l in ls)
        if all_data:
            documents = self.cross_validation_split.documents
        else:
            documents = self.cross_validation_split.train_documents
        train_documents = self.filter_documents(
            documents
        )
        self._train_set, self._early_stopping_set = split(train_documents)
        self._model = self.compile_model()
        self.epoch = 0
        self.epochs_without_improvement = 0
        self.best_score = float('inf')

    @abstractmethod
    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        raise NotImplementedError()

    @abstractmethod
    def compile_model(self) -> Model:
        raise NotImplementedError()

    @abstractmethod
    def encode_labels(self, documents: [HTMLDocument]) -> ndarray:
        raise NotImplementedError()

    @abstractmethod
    def label(self, document: HTMLDocument) -> Any:
        raise NotImplementedError()

    def is_best_score(self, metric: float) -> bool:
        return metric < self.best_score

    def recompile(self):
        log.info('recompiling %s', self.name)
        self._model = self.compile_model()
        self.epoch = 0
        self.epochs_without_improvement = 0

    def fit_early_stopping(self):
        log.info('Fitting one epoch early stopping on task: %s', self.name)
        sequences, labels = self.vocabulary.make_batch(
            self._train_set,
            self.encode_labels
        )
        self._model.fit(
            sequences,
            labels,
            epochs=self.epoch + 1,
            batch_size=16,
            initial_epoch=self.epoch
        )
        self.update_epoch()

    def update_epoch(self):
        self.epoch += 1
        early_stopping_score = self.early_stopping_score()
        if self.is_best_score(early_stopping_score):
            log.info('New best loss found. '
                     'Resetting epochs without improvement for task: %s',
                     self.name)
            self.best_score = early_stopping_score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def fit(self):
        log.info('Fitting task %s on all data for 1 epoch', self.name)
        data = self._train_set + self._early_stopping_set
        sequences, labels = self.vocabulary.make_batch(data, self.encode_labels)
        self._model.fit(sequences, labels, epochs=self.epoch + 1, batch_size=16, initial_epoch=self.epoch)
        self.epoch += 1

    def train_score(self) -> float:
        documents = self.filter_documents(
            self.cross_validation_split.train_documents
        )
        return self._score(documents)

    def _score(self, documents) -> float:
        sequences, labels = self.vocabulary.make_batch(
            documents,
            self.encode_labels
        )
        return self._model.evaluate(sequences, labels)

    def test_score(self) -> float:
        documents = self.filter_documents(
            self.cross_validation_split.test_documents
        )
        return self._score(documents)

    def early_stopping_score(self) -> float:
        return self._score(self._early_stopping_set)
