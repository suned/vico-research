import ast
import logging

import os

import numpy
import pandas
from serum import Component, inject
import pickle

from vico.console_arguments import ConsoleArguments
from vico.html_document import HTMLDocument

log = logging.getLogger('vico.read')


def parse_list(l):
    return ast.literal_eval(l) if l else None


class DataSetReader(Component):
    args = inject(ConsoleArguments)

    def read_documents(self) -> [HTMLDocument]:
        path = self.args.get().data_dir
        sample_path = os.path.join(path, 'labels.csv')
        log.info('Reading labels from %s', sample_path)
        n_samples = self.args.get().n_samples
        samples = pandas.read_csv(
            sample_path,
            converters={'tokens': parse_list, 'brand_bio_labels': parse_list},
            nrows=n_samples
        )
        docs = []
        for i, sample in samples.iterrows():
            windows, brand_bio_labels = self.make_windows(sample)
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
                windows=windows,
                brand_bio_labels=brand_bio_labels

            )
            if i == n_samples:
                break
            docs.append(doc)
        return docs

    def make_windows(self, sample):
        def encode(bio):
            return 0.0 if bio == 'O' else 1.0
        windows = []
        labels = []
        with open('data/indices.pkl', 'rb') as f:
            indices = pickle.load(f)
        window_size = self.args.get().window_size // 2
        n_tokens = len(sample.tokens)
        for i, token in enumerate(sample.tokens):
            pre_padding_tokens = -1 * (i - window_size)
            pre_padding_tokens = (pre_padding_tokens if pre_padding_tokens > 0
                                  else 0)
            pre_padding_tokens = pre_padding_tokens
            pre_padding = [0 for _ in range(pre_padding_tokens)]
            post_padding_tokens = -1 * ((n_tokens - i) - window_size - 1)
            post_padding_tokens = (post_padding_tokens
                                   if post_padding_tokens > 0
                                   else 0)
            post_padding = [0 for _ in range(post_padding_tokens)]
            start_index = i - window_size if i - window_size > 0 else 0
            end_index = i + (window_size + 1 if i + window_size + 1 < n_tokens
                             else n_tokens)
            window = [indices[t] for t in sample.tokens[
                                              start_index:end_index
                                              ]]
            window = pre_padding + window + post_padding
            label = encode(sample.brand_bio_labels[i])
            assert len(window) == self.args.get().window_size
            windows.append(window)
            labels.append(label)
        return numpy.array(windows), numpy.array(labels)
