from keras import Model

from vico.html_document import HTMLDocument
from vico.tasks import Task
from abc import abstractmethod
from numpy import ndarray

from typing import Any


class SequenceClassificationTask(Task):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def scoring_function(self) -> str:
        return 'crossentropy'

    @abstractmethod
    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        pass

    def compile_model(self) -> Model:
        pass

    def encode_labels(self, documents: [HTMLDocument]) -> ndarray:
        pass

    @abstractmethod
    def label(self, document: HTMLDocument) -> Any:
        pass

    def fit(self):
        pass

    def fit_early_stopping(self):
        pass
