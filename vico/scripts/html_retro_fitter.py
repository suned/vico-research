import pickle
from gensim.models import Word2Vec
import argparse

from serum import Environment

from vico import Config
from vico.argument_provider import ArgumentProvider
from vico.database import DocumentDatabase

parser = argparse.ArgumentParser('Retrofit word vectors for HTML tags')
parser.add_argument('database_path', help='path to sqlite database')
parser.add_argument('embedding_path', help='path to pickled embedding dictionary')
parser.add_argument(
    'indices.path',
    help='output path for pickled indices dictionary'
)

args = parser.parse_args()

ArgumentProvider.config = Config(
    database_path=args.database_path,
)
with Environment(ArgumentProvider):
    database = DocumentDatabase()
    documents = database.load_documents()
    tokens = [document.tokens for document in documents]
    model = Word2Vec(size=300, min_count=0, sg=1)
    model.build_vocab(tokens)
    model.intersect_word2vec_format('data/transformed_embedding.word2vec', lockf=1.0)
    model.train(tokens, total_examples=model.corpus_count, epochs=10)


def is_tag(token: str):
    has_tag_open = token.startswith('<') or token.startswith('</')
    has_tag_close = token.endswith('>') or token.endswith('/>')
    return has_tag_open and has_tag_close


with open(args.embedding_path, 'rb') as f:
    vocab = pickle.load(f)

indices = {}
for index, (word, vector) in enumerate(vocab.items()):
    indices[word] = index
    if is_tag(word):
        vocab[word] = model.wv[word]

with open(args.embedding_path, 'wb') as f:
    pickle.dump(vocab, f)

with open(args.indices_path, 'wb') as f:
    pickle.dump(indices, f)
