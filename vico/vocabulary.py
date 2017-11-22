from typing import Set, Dict
from f import List, Immutable, Reader
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import logging

from vico import preprocess, immutable_array
from vico.html_document import Token, Tokenization
from vico.types import Batch, Labeller, Tokenizations
from vico.config import Config
from numpy import ndarray, zeros, random

log = logging.getLogger('vico.vocabulary')


class Vocabulary(Immutable):
    tokenizations: Tokenizations

    def embedding(self, config: Config) -> ndarray:
        docs = tuple(tokenization.tokens for tokenization in self.tokenizations)
        log.info('Fitting word2vec model')
        word2vec = Word2Vec(
            sentences=docs,
            size=config.embedding_dim,
            workers=8,
            min_count=0,
            iter=15
        )
        embeddings = zeros((self.size, config.embedding_dim))
        for word, index in self.indices.items():
            embeddings[index] = word2vec.wv[word]
        embeddings[self.out_of_vocab_index] = random.normal(
            0,
            .01,
            config.embedding_dim
        )
        return embeddings

    @property
    def indices(self) -> Dict[str, int]:
        return dict(
            (t, i + 1) for (i, t) in enumerate(self.unique_tokens)
        )

    @property
    def unique_tokens(self) -> Set[Token]:
        return {token for tokenization in self.tokenizations
                for token in tokenization.tokens}

    def __len__(self):
        return self.size

    def make_batch(self, batch: Tokenizations, labeller: Labeller):
        maxlen = preprocess.maxlen(self.tokenizations)
        indices = self.indices
        out_of_vocab_index = self.out_of_vocab_index

        def index(token: Token) -> int:
            return indices.get(token, out_of_vocab_index)

        def tokens2indices(doc: Tokenization):
            return doc.tokens | index

        sequences = batch | tokens2indices
        padded_sequences = immutable_array(
            pad_sequences(
                sequences,
                maxlen=maxlen,
                value=self.padding_index
            )
        )
        labels = immutable_array(labeller(t.document) for t in batch)
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


def get(self, tokenizations: Tokenizations) -> Reader[Config, Vocabulary]:
    return Reader.pure(Vocabulary(tokenizations))



