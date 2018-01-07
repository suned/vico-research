from typing import Tuple, Any
import logging
from functools import partial
from multiprocessing.pool import Pool
import os
from f import List, Reader, compose
import nltk
from bs4 import BeautifulSoup, Tag, Doctype, NavigableString, Comment

from vico.html_document import HTMLDocument, Tokenization
from vico.types import Docs, DocIterator, Tokenizations
from vico.config import Config

log = logging.getLogger('vico.preprocess')


def _useless_tags() -> List[str]:
    return List(
        tag for tag in (
            'script',
            'style',
            'button',
            'input',
            'meta',
            'img',
            'var',
            'code',
            'embed',
            'form',
            'kbd',
            'map',
            'menu',
            'object',
            'samp',
            'textarea',
            'header',
            'footer',
            'noscript',
            'link'
        )
    )


def remove_tags(doc: HTMLDocument) -> HTMLDocument:
    nothing = ''
    soup = BeautifulSoup(doc.html, 'lxml')
    for tag in _useless_tags():
        for element in soup.find_all(tag):
            element.replace_with(nothing)
    html = str(soup)
    return doc.set_html(html)


def remove_useless_tags(docs: Docs) -> Reader[Config, Docs]:
    log.info('Removing tags: %s', ', '.join(_useless_tags()))
    pool = Pool(os.cpu_count())
    pdocs = pool.map(remove_tags, docs)
    return Reader.pure(Docs(pdoc for pdoc in pdocs))


def html_tokenize_document(use_attributes, doc: HTMLDocument) -> Tokenization:
    def format_attributes(tag: Tag) -> str:
        attributes = []
        for key, value in tag.attrs.items():
            attributes.append('{}={}'.format(key, value))
        return ' ' + ', '.join(attributes) if attributes else ''

    def format_tag(tag) -> str:
        attributes = format_attributes(tag) if use_attributes else ''
        return '<{}{}>'.format(tag.name, attributes)

    def format_endtag(tag) -> str:
        return '</{}>'.format(tag.name)

    def tokenize(text: str) -> Tuple:
        if isinstance(text, Comment):
            return ()
        return tuple(nltk.word_tokenize(text))

    def tokenize_element(element: Tag) -> Tuple:
        if isinstance(element, Doctype):
            return ()
        if isinstance(element, NavigableString):
            return tokenize(element)
        if isinstance(element, Tag):
            ts = (format_tag(element),)
            for child in element.children:
                ts += tokenize_element(child)
            ts += (format_endtag(element),)
            return ts
        raise ValueError('Unexpected type: {}'.format(type(element)))

    soup = BeautifulSoup(doc.html, 'lxml')
    tokens = tokenize_element(soup)
    return doc.set_tokens(List(t for t in tokens))


def html_tokenize(docs: Docs) -> Reader[Config, Tokenizations]:
    def _(config: Config) -> Reader[Config, Tokenizations]:
        log.info('HTML tokenizing documents')
        with Pool(os.cpu_count()) as pool:
            tokenize = partial(html_tokenize_document, config.use_attributes)
            pdocs = pool.map(tokenize, docs.values)
        return Reader.pure(List(pdoc for pdoc in pdocs))
    return Reader.ask(Config) >> _


def simple_tokenize(docs: Docs) -> Reader[Config, Tokenizations]:
    def tokenize(doc: HTMLDocument) -> Tokenization:
        tokens = doc.html.split(' ')
        return doc.set_tokens(List(t for t in tokens))
    return Reader.pure(docs | tokenize)


def to_lower(t: Tokenization) -> Tokenization:
    lower_tokens = t.tokens | (lambda token: token.lower())
    return Tokenization(t.document, lower_tokens)


def lowercase(docs: Tokenizations) -> Reader[Config, Tokenizations]:
    log.info('Lowercase documents')
    return Reader.pure(docs | to_lower)


def maxlen(tokenizations: Tokenizations) -> int:
    return max(len(t.tokens) for t in tokenizations)


def pipeline(docs: DocIterator) -> Reader[Config, Tokenizations]:
    ds = List(docs)
    log.info('Pre-processing %i documents', len(ds))
    return remove_useless_tags(ds) >> html_tokenize >> lowercase
