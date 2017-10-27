import copy
import os
from pymonad import List

Token = str
Tokens = List


class NotTokenizedError(Exception):
    pass


class HTMLDocument:
    def __init__(self,
                 html: str = '',
                 brand: str = '',
                 gtin13: int = 0,
                 ean: int = 0,
                 asin: str = '',
                 sku: int = 0,
                 price: float = 0.0,
                 currency: str = '',
                 path: str = '',
                 tokens: Tokens = None) -> None:
        self._html = html
        self._brand = brand
        self._gtin13 = gtin13
        self._ean = ean
        self._asin = asin
        self._sku = sku
        self._price = price
        self._currency = currency
        self._path = path
        self._tokens = tokens

    @property
    def html(self) -> str:
        return self._html

    def set_html(self, html):
        new_html_document = copy.deepcopy(self)
        new_html_document._html = html
        return new_html_document

    @property
    def brand(self) -> str:
        return self._brand

    @property
    def gtin13(self) -> int:
        return self._gtin13

    @property
    def ean(self) -> int:
        return self._ean

    @property
    def asin(self) -> str:
        return self._asin

    @property
    def sku(self) -> int:
        return self._sku

    @property
    def price(self) -> float:
        return self._price

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def tokens(self) -> Tokens:
        if not self._tokens:
            raise NotTokenizedError(
                'Document {} is not tokenized'.format(self._path)
            )
        return self._tokens

    def set_tokens(self, tokens):
        new_html_document = copy.deepcopy(self)
        new_html_document._tokens = tokens
        return new_html_document

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
            value_or_none(self._path)
        )
