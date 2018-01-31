import os
from typing import List

from serum import inject

from f import Immutable
from numpy import ndarray

from vico.console_arguments import ConsoleArguments

Token = str
Tokens = List[Token]


class NotTokenizedError(Exception):
    pass


class HTMLDocument(Immutable):
    html: str
    brand: str = None
    gtin13: int = None
    ean: int = None
    asin: int = None
    sku: str = None
    price: float = None
    currency: str = None
    path: str = None
    vendor: str = None
    language: str = None
    tokens: Tokens = None
    brand_bio_labels: List = None
    windows_5: List = None
    windows_11: List = None
    windows_21: List = None

    @property
    def windows(self):
        args = ConsoleArguments()
        size = args.get().window_size
        if size == 5:
            return self.windows_5
        elif size == 11:
            return self.windows_11
        return self.windows_21

    def set_html(self, html: str) -> 'HTMLDocument':
        d = self._asdict()
        d['html'] = html
        return HTMLDocument(**d)

    def set_tokens(self, tokens: List[Token]) -> 'HTMLDocument':
        d = vars(self)
        d['tokens'] = tokens
        return HTMLDocument(**d)

    def __repr__(self) -> str:
        def value_or_none(value):
            return value if value else 'None'

        def format_html():
            html = self.html.replace(os.linesep, ' ')
            return html if len(html) < 20 else html[:20] + '...'

        return '''HTMLDocument(
    html={},
    brand={},
    gtin13={},
    ean={},
    asin={},
    sku={},
    price={},
    currency={},
    path={}
)'''.format(
            format_html(),
            value_or_none(self.brand),
            value_or_none(self.gtin13),
            value_or_none(self.ean),
            value_or_none(self.asin),
            value_or_none(self.sku),
            value_or_none(self.price),
            value_or_none(self.currency),
            value_or_none(self.path)
        )
