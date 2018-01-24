from typing import Set, Dict

from serum import inject, Singleton

from keras.preprocessing.sequence import pad_sequences
import logging
import pickle

from vico import preprocess, immutable_array
from vico.console_arguments import ConsoleArguments
from vico.cross_validation_split import CrossValidationSplit
from vico.html_document import Token, HTMLDocument, HTMLDocument
from vico.types import Labeller
from numpy import ndarray, zeros, random

log = logging.getLogger('vico.vocabulary')


class Vocabulary(Singleton):
    cross_validation_split = inject(CrossValidationSplit)
    args = inject(ConsoleArguments)

    @property
    def embedding(self) -> ndarray:
        if self._embedding is None:
            self._embedding = self._read_embedding()
        return self._embedding

    def __init__(self):
        self._embedding = None

    def _read_embedding(self):
        config = self.args.get()
        with open(config.embedding_path, 'rb') as embedding_file:
            fasttext_vectors = pickle.load(embedding_file)
        embeddings = random.uniform(
            low=.01,
            high=.1,
            size=(self.size, 300)
        )
        indices = self.indices
        for word, index in indices.items():
            embeddings[index] = fasttext_vectors[word]
        embeddings[self.out_of_vocab_index] = random.normal(
            0,
            .01,
            300
        )
        embeddings[self.padding_index] = zeros(300)
        return embeddings

    @property
    def indices(self) -> Dict[str, int]:
        return dict(
            (t, i + 1) for (i, t) in enumerate(self.unique_tokens)
        )

    @property
    def unique_tokens(self) -> Set[Token]:
        return {token for doc in
                self.cross_validation_split.documents
                for token in doc.tokens}

    @property
    def languages(self):
        return {doc.language for doc in
                self.cross_validation_split.documents
                if doc.language}

    def __len__(self):
        return self.size

    def make_batch(self, batch: [HTMLDocument], labeller: Labeller):
        maxlen = preprocess.maxlen(self.cross_validation_split.documents)
        indices = self.indices
        out_of_vocab_index = self.out_of_vocab_index

        def index(token: Token) -> int:
            return indices.get(token, out_of_vocab_index)

        def tokens2indices(doc: HTMLDocument):
            return [index(token) for token in doc.tokens]

        sequences = [tokens2indices(doc) for doc in batch]
        padded_sequences = immutable_array(
            pad_sequences(
                sequences,
                maxlen=maxlen,
                value=self.padding_index
            )
        )
        labels = labeller(batch)
        return padded_sequences, labels

    @property
    def padding_index(self) -> int:
        return 0

    @property
    def out_of_vocab_index(self) -> int:
        return len(self.indices) + 1

    @property
    def size(self) -> int:
        # + 2 because of padding and out of vocab tokens
        return len(self.unique_tokens) + 2
