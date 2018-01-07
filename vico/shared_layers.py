import logging
from typing import Tuple

from keras.models import Layer
from keras.initializers import RandomUniform, glorot_uniform
from keras.layers import Embedding, Conv1D, GlobalMaxPool1D, concatenate

from vico import Config
from vico.vocabulary import Vocabulary

log = logging.getLogger('vico.shared_layers')


def get_embedding_layer(vocab: Vocabulary, config: Config) -> Layer:
    embedding = vocab.embedding(config)
    _, embedding_dim = embedding.shape
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
        weights=[embedding]
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


class SharedLayers:
    def __init__(self, vocabulary: Vocabulary, config: Config):
        self._embedding_layer = get_embedding_layer(vocabulary, config)
        self._convolutions = get_convolutional_layers(config)

    def __call__(self, input_layer):
        embedding_tensor = self._embedding_layer(input_layer)
        pooling_tensors = []
        for convolutional_layer in self._convolutions:
            convolutional_tensor = convolutional_layer(embedding_tensor)
            pooling_tensor = GlobalMaxPool1D()(convolutional_tensor)
            pooling_tensors.append(pooling_tensor)
        return concatenate(pooling_tensors)
