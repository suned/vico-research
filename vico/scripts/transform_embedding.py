import argparse
import os
from serum import Environment

from vico.argument_provider import ArgumentProvider
from vico.database import DocumentDatabase
from .. import fasttext
from ..config import Config
from itertools import groupby
from numpy import random
import pickle


def is_tag(token: str):
    has_tag_open = token.startswith('<') or token.startswith('</')
    has_tag_close = token.endswith('>') or token.endswith('/>')
    return has_tag_open and has_tag_close


parser = argparse.ArgumentParser(
    'Combine and transform pretrained word vectors into a pickled dictionary'
)
parser.add_argument('database_path', help='path to sqlite database')
parser.add_argument('embedding_path', help='output path to pickled embedding dictionary')
parser.add_argument(
    'pretrained_vector_folder',
    help='path to folder with pretrained vectors'
)
parser.add_argument(
    'transformation_matrix_folder',
    help='path to folder with transformation matrices'
)

args = parser.parse_args()

ArgumentProvider.config = Config(
    database_path=args.database_path,
)
with Environment(ArgumentProvider):
    database = DocumentDatabase()
    documents = database.load_documents()
    documents = sorted(documents, key=lambda d: d.language)
    embedding = {}
    for language, group in groupby(documents, key=lambda d: d.language):
        vector_filename = f'wiki.{language}.vec'
        vector_path = os.path.join(args.pretrained_vector_folder, vector_filename)
        lang_embedding = fasttext.FastVector(vector_path)
        transformation_matrix_filename = f'{language}.txt'
        transformation_matrix_path = os.path.join(
            args.transformation_matrix_folder,
            transformation_matrix_filename
        )
        lang_embedding.apply_transform(
            transformation_matrix_path
        )
        for document in group:
            for token in document.tokens:
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
with open(args.embedding_path, 'wb') as f:
    pickle.dump(embedding, f)
