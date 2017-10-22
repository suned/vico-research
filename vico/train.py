import logging

from keras.models import Model

from vico import model, vocabulary, preprocess, label
from vico.types import DocsIterator

log = logging.getLogger('vico.train')


def early_stopping(docs: DocsIterator) -> Model:
    vocab_size = vocabulary.size(docs)
    maxlen = preprocess.maxlen(docs)
    network = model.get(
        input_length=maxlen,
        vocab_size=vocab_size
    )
    batch_generator = vocabulary.batch_generator(docs, labeller=label.price)
    sequences, labels = batch_generator(docs)
    log.info('Fitting model')
    network.fit(sequences, labels)
    return network
