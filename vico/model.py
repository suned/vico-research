import logging
from keras.models import Model
from keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Layer,
    Embedding,
    Dense
)

from vico.config import Config

log = logging.getLogger('vico.model')


def get_input_layer(length: int) -> Input:
    log.info('Building input layer with input length: %i', length)
    return Input(
        shape=(length,),
        name='token_input'
    )


def get_embedding_layer(vocab_size: int, config: Config) -> Layer:
    embedding_dim = config.embedding_dim
    log.info(
        'Building embedding layer with vocab size: %i and embedding dimension: %i',
        vocab_size,
        embedding_dim
    )
    return Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name='embedding_layer'
    )


def get_bilstm_layer(config: Config) -> Layer:
    dim = config.bilstm_dim
    log.info('Building bi-LSTM layer with hidden dimension: %i', dim)
    return Bidirectional(LSTM(dim), name='bilstm_layer')


def get_output_layer() -> Layer:
    log.info('Building output layer')
    return Dense(1, activation='linear', name='output_layer')


def get(input_length: int, vocab_size: int, config: Config) -> Model:
    log.info('Building model')
    input_layer = get_input_layer(input_length)
    embedding_layer = get_embedding_layer(vocab_size, config)(input_layer)
    recurrent_layer = get_bilstm_layer(config)(embedding_layer)
    output = get_output_layer()(recurrent_layer)
    model = Model(
        inputs=input_layer,
        outputs=output,
    )
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    return model
