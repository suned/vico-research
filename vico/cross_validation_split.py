from abc import abstractmethod
from typing import Tuple

from numpy.core.multiarray import ndarray
from pandas import Series
from serum import Singleton, inject
from sklearn.model_selection import LeaveOneGroupOut

from vico.console_arguments import ConsoleArguments
from vico.data_set_reader import DataSetReader
from vico.html_document import HTMLDocument


class CrossValidationSplit(Singleton):
    reader = inject(DataSetReader)
    args = inject(ConsoleArguments)

    @property
    def train_documents(self) -> [HTMLDocument]:
        if self._train_indices is None:
            raise ValueError('No current split')
        doc_series = Series(self.documents)
        train_docs = doc_series[self._train_indices]
        return list(train_docs)

    @property
    def test_documents(self) -> [HTMLDocument]:
        if self._test_indices is None:
            raise ValueError('No current split')
        doc_series = Series(self.documents)
        test_docs = doc_series[self._test_indices]
        return list(test_docs)

    @abstractmethod
    def split(self) -> [Tuple[ndarray]]:
        raise NotImplementedError()

    def __len__(self):
        return len(self._splits)

    @property
    def documents(self) -> [HTMLDocument]:
        if self._documents is None:
            self._documents = self.reader.read_documents()
        return self._documents

    def __next__(self):
        try:
            self._train_indices, self._test_indices = self._splits.pop()
            if self.skip():
                return next(self)
            return self._train_indices, self._test_indices
        except IndexError:
            raise StopIteration()

    def __iter__(self):
        return self

    def __init__(self):
        self._documents: [HTMLDocument] = None
        self._train_indices: ndarray = None
        self._test_indices: ndarray = None
        self._splits = self.split()

    def skip(self):
        test_doc = self.test_documents[0]
        return test_doc.language in self.args.get().skip


class LeaveOneLanguageOut(CrossValidationSplit):
    def split(self) -> [Tuple[ndarray]]:
        documents = Series(self.documents)
        languages = [doc.language for doc in documents]
        brands = [doc.brand for doc in documents]
        splits = LeaveOneGroupOut().split(documents, brands, languages)
        return list(splits)
