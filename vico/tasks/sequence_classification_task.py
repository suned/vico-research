import numpy
from keras import Model, Input
from keras.layers import Dense
from serum import inject

from vico.console_arguments import ConsoleArguments
from vico.html_document import HTMLDocument
from .task import Task
from abc import abstractmethod
from numpy import ndarray

from typing import Any


class SequenceClassificationTask(Task):
    config = inject(ConsoleArguments)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def scoring_function(self) -> str:
        return 'sigmoid'

    @abstractmethod
    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        pass

    def compile_model(self) -> Model:
        window_size = self.config.get().window_size
        input_layer = Input(
            shape=(window_size,)
        )
        shared_tensor = self._shared_layers(input_layer)
        output = Dense(
            1,
            activation='sigmoid'
        )(shared_tensor)
        model = Model(
            inputs=input_layer,
            outputs=output
        )
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy'
        )
        return model

    def encode_labels(self, documents: [HTMLDocument]) -> ndarray:
        return numpy.array([])

    def label(self, document: HTMLDocument) -> Any:
        return 0.0

    def fit(self):
        train_windows = [d.windows for d in self._train_set]
        train_labels = [d.brand_bio_labels for d in self._train_set]
        test_windows = [d.windows for d in self._early_stopping_set]
        test_labels = [d.brand_bio_labels for d in self._early_stopping_set]
        data = numpy.vstack(
            train_windows + test_windows
        )
        labels = numpy.concatenate(
            train_labels + test_labels
        )
        self._model.fit(
            data,
            labels,
            epochs=self.epoch + 1,
            batch_size=16,
            initial_epoch=self.epoch
        )

    def train_score(self):
        windows = [d.windows for d in self._train_set]
        labels = [d.brand_bio_labels for d in self._train_set]
        windows = numpy.vstack(windows)
        labels = numpy.concatenate(labels)
        return self._score(
            windows,
            labels
        )

    def early_stopping_score(self):
        windows = [d.windows for d in self._early_stopping_set]
        labels = [d.brand_bio_labels for d in self._early_stopping_set]
        windows = numpy.vstack(windows)
        labels = numpy.concatenate(labels)
        return self._score(
            windows,
            labels
        )

    def _score(self, windows, labels) -> float:
        return self._model.evaluate(
            windows,
            labels
        )

    def test_score(self):
        windows = [d.windows for d in self.cross_validation_split.test_documents]
        labels = [d.brand_bio_labels for d in self.cross_validation_split.test_documents]
        windows = numpy.vstack(windows)
        labels = numpy.concatenate(labels)
        return self._score(
            windows,
            labels
        )

    def fit_early_stopping(self):
        windows = [d.windows for d in self._early_stopping_set]
        labels = [d.brand_bio_labels for d in self._early_stopping_set]
        windows = numpy.vstack(windows)
        labels = numpy.concatenate(labels)
        self._model.fit(
            windows,
            labels,
            epochs=self.epoch + 1,
            batch_size=16,
            initial_epoch=self.epoch
        )
        self.update_epoch()
