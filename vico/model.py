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

from vico import args

log = logging.getLogger('vico.model')



def get_input_layer(length: int) -> Input:
    log.info('Building input layer with input length: %i', length)
    return Input(
        shape=(length,),
        name='token_input'
    )


def get_embedding_layer(vocab_size: int) -> Layer:
    embedding_dim = args.get().embedding_dim
    log.info(
        'Building embedding layer with vocab size: %i and embedding dim: %i',
        vocab_size,
        embedding_dim
    )
    return Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name='embedding_layer'
    )


def get_recurrent_layer() -> Layer:
    dim = args.get().bilstm_dim
    log.info('Building recurrent layer with hidden dimension: %i', dim)
    return Bidirectional(LSTM(args.get().bilstm_dim), name='bilstm_layer')


def get_output_layer() -> Layer:
    log.info('Building output layer')
    return Dense(1, activation='linear', name='output_layer')


def get(input_length: int, vocab_size: int) -> Model:
    log.info('Building model')
    input_layer = get_input_layer(input_length)
    embedding_layer = get_embedding_layer(vocab_size)(input_layer)
    recurrent_layer = get_recurrent_layer()(embedding_layer)
    output = get_output_layer()(recurrent_layer)
    model = Model(
        input=input_layer,
        output=output,
    )
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    return model
