import ast
import logging

import os

import pandas
from serum import Component, inject

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
                brand_bio_labels=sample.brand_bio_labels
            )
            if i == n_samples:
                break
            docs.append(doc)
        return docs
