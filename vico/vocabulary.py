from keras.preprocessing.sequence import pad_sequences

from vico import preprocess
from vico.types import DocsIterator, BatchGenerator, Batch
from vico.pipes import as_array


def padding():
    return 0


def size(docs):
    # + 2 because of padding and out of vocab tokens
    return len(_unique_tokens(docs)) + 2


def batch_generator(docs: DocsIterator) -> BatchGenerator:
    maxlen = preprocess.maxlen(docs)
    tokens = _unique_tokens(docs)
    # i + 1 because we use 0 to pad short sequences
    vocab = dict((t, i + 1) for (i, t) in enumerate(tokens))

    def tokens2indices(ds: DocsIterator) -> Batch:
        out_of_vocab = len(vocab) + 1
        sequences = []
        for doc in ds:
            indices = [vocab.get(t, out_of_vocab) for t in doc.tokens]
            sequences.append(indices)
        padded_sequences = pad_sequences(sequences, maxlen=maxlen, value=padding())
        labels = (doc.price for doc in ds) | as_array
        return padded_sequences, labels

    return tokens2indices


def _unique_tokens(docs):
    return {t for d in docs for t in d.tokens}
