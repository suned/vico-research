from typing import Tuple, Any
import logging
from functools import partial
from multiprocessing.pool import Pool
import os
from f import List, Reader, compose
import nltk
from bs4 import BeautifulSoup, Tag, Doctype, NavigableString, Comment

from vico.html_document import HTMLDocument, HTMLDocument
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


def remove_useless_tags(docs: [HTMLDocument]) -> [HTMLDocument]:
    log.info('Removing tags: %s', ', '.join(_useless_tags()))
    with Pool(os.cpu_count()) as pool:
        return pool.map(remove_tags, docs)


def html_tokenize_document(doc: HTMLDocument) -> HTMLDocument:
    use_attributes = False

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


def html_tokenize(docs: [HTMLDocument]) -> [HTMLDocument]:
    log.info('HTML tokenizing documents')
    with Pool(os.cpu_count()) as pool:
        return pool.map(html_tokenize_document, docs)


def simple_tokenize(docs: Docs) -> Reader[Config, Tokenizations]:
    def tokenize(doc: HTMLDocument) -> HTMLDocument:
        tokens = doc.html.split(' ')
        return doc.set_tokens(List(t for t in tokens))
    return Reader.pure(docs | tokenize)


def lowercase(docs: [HTMLDocument]) -> [HTMLDocument]:
    log.info('Lowercase documents')
    return [doc.set_tokens([t.lower() for t in doc.tokens]) for doc in docs]


def maxlen(documents: [HTMLDocument]) -> int:
    return max(len(doc.tokens) for doc in documents)


def pipeline(docs: DocIterator) -> Reader[Config, Tokenizations]:
    ds = List(docs)
    log.info('Pre-processing %i documents', len(ds))
    return remove_useless_tags(ds) >> html_tokenize >> lowercase
