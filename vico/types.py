from typing import Callable, Iterator, Dict, Tuple
from numpy import ndarray

from vico.html_document import HTMLDocument

DocsIterator = Iterator[HTMLDocument]
DocsGenerator = Callable[[], DocsIterator]
Docs = Tuple[HTMLDocument]
Vocabulary = Dict[str, int]
Batch = Tuple[ndarray, ndarray]
BatchGenerator = Callable[[DocsIterator], Batch]