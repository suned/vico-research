from typing import Callable, Dict, Tuple, Any, Generator, Iterator
from pymonad import List

from numpy import ndarray

from vico.html_document import HTMLDocument


Docs = List
Batch = Tuple[ndarray, ndarray]
Batcher = Callable[[Docs], Batch]
DocIterator = Iterator[HTMLDocument]
Vocabulary = Dict[str, int]
Labeller = Callable[[HTMLDocument], Any]
Fold = Tuple[Docs, Docs]
Folds = Generator[Fold, None, None]

