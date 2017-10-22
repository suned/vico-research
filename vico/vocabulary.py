from keras.preprocessing.sequence import pad_sequences

from vico import preprocess
from vico.types import DocsIterator, BatchGenerator, Batch, Labeller
from vico.pipes import as_array


def padding():
    return 0


def size(docs: DocsIterator) -> int:
    # + 2 because of padding and out of vocab tokens
    return len(_unique_tokens(docs)) + 2


def batch_generator(docs: DocsIterator, labeller: Labeller) -> BatchGenerator:
    maxlen = preprocess.maxlen(docs)
    vocab = get(docs)

    def generate(batch: DocsIterator) -> Batch:
        out_of_vocab = len(vocab) + 1
        sequences = []
        for doc in batch:
            indices = [vocab.get(t, out_of_vocab) for t in doc.tokens]
            sequences.append(indices)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=maxlen,
            value=padding()
        )
        labels = (labeller(doc) for doc in docs) | as_array
        return padded_sequences, labels

    return generate


def get(docs):
    tokens = _unique_tokens(docs)
    # i + 1 because we use 0 to pad short sequences
    return dict((t, i + 1) for (i, t) in enumerate(tokens))


def _unique_tokens(docs):
    return {t for d in docs for t in d.tokens}
