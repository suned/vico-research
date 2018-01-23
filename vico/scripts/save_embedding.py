from .. import read, preprocess, fasttext, configure_root_logger
from ..validate import limit
from ..config import Config
from itertools import groupby
from numpy import random
import pickle
import logging


def is_tag(token: str):
    has_tag_open = token.startswith('<') or token.startswith('</')
    has_tag_close = token.endswith('>') or token.endswith('/>')
    return has_tag_open and has_tag_close


config = Config(log_level=logging.INFO)
configure_root_logger(config)
tokenizations = read.from_csv()(config)
import ipdb
ipdb.sset_trace()
tokenizations = sorted(tokenizations, key=lambda t: t.document.language)
embedding = {}
for language, group in groupby(tokenizations, key=lambda t: t.document.language):
    lang_embedding = fasttext.FastVector('data/pretrained_word_vectors/wiki.' + language + '.vec')
    lang_embedding.apply_transform(
        'data/transformation_matrices/' + language + '.txt'
    )
    for tokenization in group:
        for token in tokenization.tokens:
            if token in lang_embedding:
                embedding[token] = lang_embedding[token]
            if token in embedding:
                continue
            else:
                vector = random.normal(
                    0,
                    .01,
                    300
                )
                embedding[token] = vector
with open('data/transformed_embedding.pkl', 'wb') as f:
    pickle.dump(embedding, f)


