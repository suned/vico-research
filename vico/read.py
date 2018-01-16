import os
from f import Reader
from pandas import DataFrame, Series, read_csv, to_numeric
import logging
import re
from urllib.parse import urlparse

from vico.config import Config
from vico.html_document import HTMLDocument
from vico.types import DocIterator

log = logging.getLogger('vico.read')


def _vendor(url):
    parse = urlparse(url)
    pattern = '(www\.)([a-z\-]*)(\.[a-z]*)'
    m = re.match(pattern, parse.hostname)
    return m[2] if m else None


def all_docs() -> Reader[Config, DocIterator]:
    def _(config: Config) -> Reader[Config, DocIterator]:
        path = config.data_dir

        def get_file_path(s: Series) -> str:
            filename = os.path.split(s.local_path)[-1]
            return os.path.join(path, 'pages', filename)

        def read_html(s: Series) -> str:
            file_path = get_file_path(s)
            with open(file_path) as f:
                return f.read()

        def read_samples() -> DataFrame:
            labels_path = os.path.join(path, 'CosmeticsTestData.csv')
            labels = read_csv(labels_path, sep=';')
            cleaned_url = (labels.url.str
                           .replace('http://', '')
                           .str.replace('https://', '')
                           .str.replace('www.', 'www_')
                           .str.replace('.aspx', 'aspx')
                           .str.replace('/', '_')
                           .str.replace('.', '_')
                           .str.lower())
            index = labels.group.astype(str) + '-' + cleaned_url + '.html'
            labels.index = index
            labels.price = to_numeric(
                labels.price.str.replace(',', '.'),
                errors='coerce'
            )
            return labels.dropna(subset=['price'])

        def generate() -> DocIterator:
            log.info('Reading data from %s', path)
            samples = read_samples()
            not_found_count = 0
            for name, sample in samples.iterrows():
                try:
                    html = read_html(sample)
                    yield HTMLDocument(
                        html=html,
                        brand=sample.brand,
                        gtin13=sample.gtin13,
                        ean=sample.ean,
                        asin=sample.asin,
                        sku=sample.sku,
                        price=sample.price,
                        currency=sample.currency,
                        path=get_file_path(sample),
                        vendor=_vendor(sample.url)
                    )
                except FileNotFoundError:
                    log.debug('File not found: %s', get_file_path(sample))
                    not_found_count += 1
            log.warning('Could not locate %i documents', not_found_count)
        return Reader.pure(generate())
    return Reader.ask(Config) >> _
