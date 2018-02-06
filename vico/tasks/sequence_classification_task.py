import numpy
from keras import Model, Input
from keras.layers import Dense
from serum import inject

from vico.console_arguments import ConsoleArguments
from vico.html_document import HTMLDocument
from .task import Task
from abc import abstractmethod
from numpy import ndarray


class SequenceClassificationTask(Task):
    config = inject(ConsoleArguments)

    def stack(self, documents):
        windows = [d.windows for d in
                   documents]
        labels = [self.label(d) for d in
                  documents]
        windows = numpy.vstack(windows)
        labels = numpy.concatenate(labels)
        return windows, labels

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def label(self, document):
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

    def fit(self):
        windows, labels = self.stack(self._train_set + self._early_stopping_set)
        self._model.fit(
            windows,
            labels,
            epochs=self.epoch + 1,
            batch_size=16,
            initial_epoch=self.epoch
        )
        self.epoch += 1

    def train_score(self):
        windows, labels = self.stack(self._train_set)
        return self._score(
            windows,
            labels
        )

    def early_stopping_score(self):
        windows, labels = self.stack(self._early_stopping_set)
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
        windows, labels = self.stack(self.cross_validation_split.test_documents)
        return self._score(
            windows,
            labels
        )

    def fit_early_stopping(self):
        windows, labels = self.stack(self._train_set)
        self._model.fit(
            windows,
            labels,
            epochs=self.epoch + 1,
            batch_size=16,
            initial_epoch=self.epoch
        )
        self.update_epoch()
