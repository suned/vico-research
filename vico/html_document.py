import os
from f import List, Immutable

Token = str
Tokens = List[Token]


class NotTokenizedError(Exception):
    pass


class HTMLDocument(Immutable):
    html: str
    brand: str
    gtin13: int
    ean: int
    asin: int
    sku: int
    price: float
    currency: str
    path: str

    def set_html(self, html: str) -> 'HTMLDocument':
        d = self._asdict()
        d['html'] = html
        return HTMLDocument(**d)

    def set_tokens(self, tokens: List[Token]) -> 'Tokenization':
        return Tokenization(document=self, tokens=tokens)

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


class Tokenization(Immutable):
    document: HTMLDocument
    tokens: List[Token]
