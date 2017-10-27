from typing import Set

from keras.preprocessing.sequence import pad_sequences

from vico import preprocess, immutable_array
from vico.html_document import Token, HTMLDocument
from vico.types import Batch, Labeller, Batcher, Vocabulary, Docs


def padding_index() -> int:
    return 0


def out_of_vocab_index(vocab: Vocabulary) -> int:
    return len(vocab) + 1


def size(docs: Docs) -> int:
    # + 2 because of padding and out of vocab tokens
    return len(_unique_tokens(docs)) + 2


def batcher(docs: Docs, labeller: Labeller) -> Batcher:
    maxlen = preprocess.maxlen(docs)
    vocab = get(docs)
    out_of_vocab = out_of_vocab_index(vocab)

    def index(token: Token) -> int:
        return vocab.get(token, out_of_vocab)

    def indices(doc: HTMLDocument):
        return index * doc.tokens

    def _(batch: Docs) -> Batch:
        sequences = indices * batch
        padded_sequences = immutable_array(
            pad_sequences(
                sequences,
                maxlen=maxlen,
                value=padding_index()
            )
        )

        labels = immutable_array([labeller(doc) for doc in batch])
        return padded_sequences, labels

    return _


def get(docs) -> Vocabulary:
    tokens = _unique_tokens(docs)
    # i + 1 because we use 0 to pad short sequences
    return dict((t, i + 1) for (i, t) in enumerate(tokens))


def _unique_tokens(docs) -> Set[Token]:
    return {t for d in docs for t in d.tokens}
