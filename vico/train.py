import logging

from keras.models import Model

from vico import model, vocabulary, preprocess, label, Config
from pymonad import List

log = logging.getLogger('vico.train')


def early_stopping(docs: List, config: Config) -> Model:
    vocab_size = vocabulary.size(docs)
    maxlen = preprocess.maxlen(docs)
    network = model.get(
        input_length=maxlen,
        vocab_size=vocab_size,
        config=config
    )
    batcher = vocabulary.batcher(docs, labeller=label.price)
    sequences, labels = batcher(docs)
    log.info('Fitting model')
    network.fit(sequences, labels, epochs=config.epochs)
    return network
