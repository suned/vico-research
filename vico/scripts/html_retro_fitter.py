import pandas
import pickle
from gensim.models import Word2Vec
import ast

import ipdb
ipdb.sset_trace()
labels = pandas.read_csv('data/labels.csv', converters={'tokens': lambda v: ast.literal_eval(v) if v else []})
model = Word2Vec(size=300, min_count=0, sg=1)
model.build_vocab(labels.tokens)
model.intersect_word2vec_format('data/transformed_embedding.word2vec', lockf=1.0)
model.train(labels.tokens, total_examples=model.corpus_count, epochs=10)


def is_tag(token: str):
    has_tag_open = token.startswith('<') or token.startswith('</')
    has_tag_close = token.endswith('>') or token.endswith('/>')
    return has_tag_open and has_tag_close


with open('data/transformed_embedding.pkl', 'rb') as f:
    vocab = pickle.load(f)

indices = {}
for index, (word, vector) in enumerate(vocab.items()):
    indices[word] = index
    if is_tag(word):
        vocab[word] = model.wv[word]

with open('data/transformed_embedding.pkl', 'wb') as f:
    pickle.dump(vocab, f)

with open('data/indices.pkl', 'wb') as f:
    pickle.dump(indices, f)
