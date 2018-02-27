import argparse
import pickle
import ast
import pandas

from serum import Environment

from vico.database import DocumentDatabase
from vico.html_document import HTMLDocument

parser = argparse.ArgumentParser("""
Create windows with brand and ean BIO labels
""")
parser.add_argument('indices_path', help='path to pickled indices dictionary')
parser.add_argument('database_path', help='path to sqlite database')

args = parser.parse_args()

with open(args.indices_path, 'rb') as f:
    indices = pickle.load(f)


def make_windows(sample, window_size):
    def encode(bio):
        return 0.0 if bio == 'O' else 1.0

    windows = []
    labels = []
    n_tokens = len(sample.tokens)
    half_window_size = window_size // 2
    for i, token in enumerate(sample.tokens):
        pre_padding_tokens = -1 * (i - half_window_size)
        pre_padding_tokens = (pre_padding_tokens if pre_padding_tokens > 0
                              else 0)
        pre_padding_tokens = pre_padding_tokens
        pre_padding = [0 for _ in range(pre_padding_tokens)]
        post_padding_tokens = -1 * ((n_tokens - i) - half_window_size - 1)
        post_padding_tokens = (post_padding_tokens
                               if post_padding_tokens > 0 else 0)
        post_padding = [0 for _ in range(post_padding_tokens)]
        start_index = i - half_window_size if i - half_window_size > 0 else 0
        end_index = i + (half_window_size + 1 if i + half_window_size + 1 < n_tokens
                         else n_tokens)
        window = [indices[t] for t in sample.tokens[
                                      start_index:end_index
                                      ]]
        window = pre_padding + window + post_padding
        label = encode(sample.brand_bio_labels[i])
        assert len(window) == window_size
        windows.append(window)
        labels.append(label)
    return windows, labels


samples = pandas.read_csv(
    'data/labels.csv',
    converters={
        'tokens': lambda v: ast.literal_eval(v) if v else [],
        'brand_bio_labels': lambda v: ast.literal_eval(v) if v else [],
    }
)
samples = samples.where(samples.notnull(), None)
docs = []


def is_tag(token: str):
    has_tag_open = token.startswith('<') or token.startswith('</')
    has_tag_close = token.endswith('>') or token.endswith('/>')
    return has_tag_open and has_tag_close


def remove_html(sample):
    tokens = []
    labels = []
    for token, bio_label in zip(sample.tokens, sample.brand_bio_labels):
        if is_tag(token):
            continue
        tokens.append(token)
        labels.append(bio_label)
    sample.tokens = tokens
    sample.brand_bio_labels = labels
    return sample


for _, sample in samples.iterrows():
    if not sample.brand_bio_labels:
        continue
    sample = remove_html(sample)
    windows_5, brand_bio_labels = make_windows(sample, 5)
    windows_11, _ = make_windows(sample, 11)
    windows_21, _ = make_windows(sample, 21)
    doc = HTMLDocument(
        html='',
        brand=sample.brand,
        gtin13=sample.gtin13,
        ean=sample.ean,
        asin=sample.asin,
        sku=sample.sku,
        price=sample.price,
        currency=sample.currency,
        path=sample.path,
        vendor=sample.vendor,
        language=sample.language,
        tokens=sample.tokens,
        windows_5=windows_5,
        windows_11=windows_11,
        windows_21=windows_21,
        brand_bio_labels=brand_bio_labels
    )
    docs.append(doc)

with Environment():
    database = DocumentDatabase()
    database.save_documents(docs)
