import os
import pandas
import logging

from vico import args
from vico.html_document import HTMLDocument
from vico.types import DocsIterator

log = logging.getLogger('vico.read')


def all_docs() -> DocsIterator:
    path = args.get().data_dir

    def get_file_path(s: pandas.Series) -> str:
        filename = os.path.split(s.local_path)[-1]
        return os.path.join(path, 'pages', filename)

    def read_html(s: pandas.Series) -> str:
        file_path = get_file_path(s)
        with open(file_path) as f:
            return f.read()

    def read_samples() -> pandas.DataFrame:
        labels_path = os.path.join(path, 'CosmeticsTestData.csv')
        labels = pandas.read_csv(labels_path, sep=';')
        cleaned_url = (labels.url.str
                       .replace('http://', '')
                       .str.replace('https://', '')
                       .str.replace('www.', 'www_')
                       .str.replace('.aspx', 'aspx')
                       .str.replace('/', '_')
                       .str.replace('.', '_')
                       .str.lower())
        labels.index = labels.group.astype(str) + '-' + cleaned_url + '.html'
        labels.price = pandas.to_numeric(
            labels.price.str.replace(',', '.'),
            errors='coerce'
        )
        return labels.dropna(subset=['price'])

    log.info('Reading data from %s', path)

    samples = read_samples()
    for name, sample in samples.iterrows():
        try:
            html = read_html(sample)
            yield HTMLDocument(
                html,
                sample.brand,
                sample.gtin13,
                sample.ean,
                sample.asin,
                sample.sku,
                sample.price,
                sample.currency,
                get_file_path(sample)
            )
        except FileNotFoundError:
            log.warning('File not found: %s', get_file_path(sample))
