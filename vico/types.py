from typing import Dict, Tuple, Any, Generator, Iterator
from f import List, Unary

from numpy import ndarray

from vico.html_document import HTMLDocument, HTMLDocument


Docs = List[HTMLDocument]
Tokenizations = List[HTMLDocument]
Batch = Tuple[ndarray, ndarray]
Batcher = Unary[Tokenizations, Batch]
DocIterator = Iterator[HTMLDocument]
Vocabulary = Dict[str, int]
Labeller = Unary[HTMLDocument, ndarray]
Fold = Tuple[Tokenizations, Tokenizations]
Folds = Generator[Fold, None, None]
