import os
from f import Reader, List
from pandas import DataFrame, Series, read_csv, to_numeric
import logging


from vico.config import Config
from vico.html_document import HTMLDocument, Tokenization
from vico.types import DocIterator, Tokenizations
import ast

log = logging.getLogger('vico.read')


def from_csv() -> Reader[Config, Tokenizations]:
    def _(config: Config) -> Reader[Config, Tokenizations]:
        path = config.data_dir
        sample_path = os.path.join(path, 'labels.csv')
        samples = read_csv(
            sample_path,
            converters={'tokens': lambda v: ast.literal_eval(v) if v else None}
        )
        ts = []
        for _, sample in samples.iterrows():
            with open(sample.path) as f:
                html = f.read()
            doc = HTMLDocument(
                html=html,
                brand=sample.brand,
                gtin13=sample.gtin13,
                ean=sample.ean,
                asin=sample.asin,
                sku=sample.sku,
                price=sample.price,
                currency=sample.currency,
                path=sample.path,
                vendor=sample.vendor,
                language=sample.language
            )
            t = Tokenization(
                document=doc,
                tokens=List(sample.tokens)
            )
            ts.append(t)
        return Reader.pure(List(ts))
    return Reader.ask(Config) >> _


def all_docs() -> Reader[Config, DocIterator]:
    def _(config: Config) -> Reader[Config, DocIterator]:
        path = config.data_dir

        def read_html(s: Series) -> str:
            file_path = s.path
            with open(file_path) as f:
                return f.read()

        def read_samples() -> DataFrame:
            labels_path = os.path.join(path, 'labels.csv')
            labels = read_csv(labels_path)
            labels.price = to_numeric(
                labels.price.str.replace(',', '.'),
                errors='coerce'
            )
            return labels.dropna(subset=['path']).drop_duplicates(subset='path')

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
                        path=sample.path,
                        vendor=sample.vendor,
                        language=sample.language
                    )
                except FileNotFoundError:
                    log.debug('File not found: %s', sample.path)
                    not_found_count += 1
            log.warning('Could not locate %i documents', not_found_count)
        return Reader.pure(generate())
    return Reader.ask(Config) >> _
