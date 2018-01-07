import argparse
from functools import partial
from multiprocessing.pool import Pool
from typing import Set, Dict
from f import List, Immutable, Reader, compose
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import logging
import os
import pickle

from vico import preprocess, immutable_array
from vico.html_document import Token, Tokenization, HTMLDocument
from vico.preprocess import html_tokenize_document, to_lower
from vico.types import Batch, Labeller, Tokenizations
from vico.config import Config
from numpy import ndarray, zeros, random

log = logging.getLogger('vico.vocabulary')


class Vocabulary(Immutable):
    tokenizations: Tokenizations

    def embedding(self, config: Config) -> ndarray:
        with open(config.embedding_path, 'rb') as embedding_file:
            word2vec = pickle.load(embedding_file)
        embeddings = random.uniform(
            low=.01,
            high=.1,
            size=(self.size, word2vec.vector_size)
        )
        for word, index in self.indices.items():
            if word in word2vec.wv:
                embeddings[index] = word2vec.wv[word]
        embeddings[self.out_of_vocab_index] = random.normal(
            0,
            .01,
            word2vec.vector_size
        )
        embeddings[self.padding_index] = zeros(word2vec.vector_size)
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


def get(tokenizations: Tokenizations) -> Reader[Config, Vocabulary]:
    return Reader.pure(Vocabulary(tokenizations))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument(
        '--output-path',
        type=str,
        default='embeddings/<lang>-embeddings-<dim>.pkl'
    )
    parser.add_argument('--use-attributes', type=bool, default=False)
    return parser.parse_args()


def pipeline(use_attributes, document):
    tokenization = html_tokenize_document(use_attributes, document)
    return to_lower(tokenization)


def create_embeddings(path, use_attributes, output_path):
    def read_file(fpath):
        if fpath.endswith('.html'):
            with open(file_path) as f:
                return HTMLDocument(html=f.read())

    log.info('Reading documents')
    documents = []
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            document = read_file(file_path)
            documents.append(document)

    log.info('Tokenizing documents')
    f = partial(pipeline, use_attributes)
    with Pool(os.cpu_count()) as pool:
        tokenizations = pool.map(f, documents)
    docs = tuple(tokenization.tokens for tokenization in tokenizations)
    language = path.split(os.path.sep).pop()
    dims = [50, 100, 150, 200, 250, 300]
    for embedding_dim in dims:
        log.info('Fitting embedding with dimension %i', embedding_dim)
        word2vec = Word2Vec(
            sentences=docs,
            size=embedding_dim,
            workers=8,
            min_count=0,
            iter=15
        )
        log.info('Saving embedding')
        file_path = output_path.replace(
            '<dim>', str(embedding_dim)
        ).replace('<lang>', language)
        with open(file_path, 'wb') as f:
            pickle.dump(word2vec, f)


if __name__ == '__main__':
    args = parse_args()
    create_embeddings(**vars(args))


