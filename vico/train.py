import logging

from keras.models import Model
from keras.callbacks import EarlyStopping
from vico import model, preprocess, label, Config
from vico.types import Tokenizations
from vico.vocabulary import Vocabulary

log = logging.getLogger('vico.train')


def early_stopping(tokenizations: Tokenizations,
                   vocabulary: Vocabulary,
                   config: Config) -> Model:
    maxlen = preprocess.maxlen(tokenizations)
    network = model.get(
        input_length=maxlen,
        vocabulary=vocabulary,
        config=config
    )
    sequences, labels = vocabulary.make_batch(tokenizations, labeller=label.price)
    log.info('Fitting model with early stopping')
    es = EarlyStopping(patience=config.patience)
    network.fit(
        sequences,
        labels,
        validation_split=.2,
        epochs=config.epochs,
        callbacks=[es]
    )
    network = model.get(
        input_length=maxlen,
        vocabulary=vocabulary,
        config=config
    )
    last_epoch_with_improvement = es.stopped_epoch - config.patience
    log.info('Fitting model with optimal epoch %i', last_epoch_with_improvement)
    network.fit(
        sequences,
        labels,
        epochs=last_epoch_with_improvement
    )
    return network
