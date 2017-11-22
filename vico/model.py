import logging
from typing import Tuple
from keras.models import Model
from keras.initializers import RandomUniform, glorot_uniform
from keras.layers import (
    Input,
    Layer,
    Embedding,
    Dense,
    Conv1D,
    GlobalMaxPool1D,
    concatenate
)

from vico.config import Config
from vico.vocabulary import Vocabulary

log = logging.getLogger('vico.model')


def get_input_layer(length: int) -> Input:
    log.info('Building input layer with input length: %i', length)
    return Input(
        shape=(length,),
        name='token_input'
    )


def get_embedding_layer(vocab: Vocabulary, config: Config) -> Layer:
    embedding_dim = config.embedding_dim
    log.info(
        'Building embedding layer with vocab size: %i and embedding dimension: %i',
        vocab.size,
        embedding_dim
    )
    return Embedding(
        input_dim=vocab.size,
        output_dim=embedding_dim,
        mask_zero=False,
        name='embedding_layer',
        embeddings_initializer=RandomUniform(seed=config.seed),
        weights=[vocab.embedding(config)]
    )


def get_convolutional_layers(config: Config) -> Tuple[Layer]:
    log.info('Building CNN layers with %i filters each', config. filters)
    layers = ()
    for filter_size in config.filter_sizes:
        layer = Conv1D(
            filters=config.filters,
            kernel_size=filter_size,
            kernel_initializer=glorot_uniform(config.seed),
            activation='relu'
        )
        layers += (layer,)
    return layers


def get_output_layer() -> Layer:
    log.info('Building output layer')
    return Dense(1, activation='linear', name='output_layer')


def get(input_length: int, vocabulary: Vocabulary, config: Config) -> Model:
    log.info('Building model')
    input_layer = get_input_layer(input_length)
    embedding_layer = get_embedding_layer(vocabulary, config)(input_layer)
    c_layers = []
    for layer in get_convolutional_layers(config):
        layer = layer(embedding_layer)
        layer = GlobalMaxPool1D()(layer)
        c_layers.append(layer)
    c_layers = concatenate(c_layers, name='pooling_concatenation')
    output = get_output_layer()(c_layers)
    model = Model(
        inputs=input_layer,
        outputs=output,
    )
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    return model
