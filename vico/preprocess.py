import logging
from functools import partial
from multiprocessing.pool import Pool
import os
from typing import List

import nltk
from bs4 import BeautifulSoup, Tag, Doctype, NavigableString

from vico.html_document import HTMLDocument
from vico.types import DocsIterator

log = logging.getLogger('vico.preprocess')


def _useless_tags() -> List[str]:
    return [
        'script',
        'style',
        'button',
        'input',
        'meta',
        'noscript',
        'link'
    ]


def _remove_tags(doc: HTMLDocument) -> HTMLDocument:
    nothing = ''
    soup = BeautifulSoup(doc.html, 'lxml')
    for tag in _useless_tags():
        for element in soup.find_all(tag):
            element.replace_with(nothing)
    html = str(soup)
    return doc.set_html(html)


def remove_useless_tags(docs: DocsIterator) -> DocsIterator:
    log.info('Removing tags: %s', ', '.join(_useless_tags()))
    pool = Pool(os.cpu_count())
    return pool.map(_remove_tags, docs)


def _html_tokenize(doc: HTMLDocument, use_attributes=False) -> HTMLDocument:
    def format_attributes(tag):
        attributes = []
        for key, value in tag.attrs.items():
            attributes.append('{}={}'.format(key, value))
        return ' ' + ', '.join(attributes) if attributes else ''

    def format_tag(tag):
        attributes = format_attributes(tag) if use_attributes else ''
        return '<{}{}>'.format(tag.name, attributes)

    def format_endtag(tag):
        return '</{}>'.format(tag.name)

    def tokenize(text):
        return tuple(nltk.word_tokenize(text))

    def tokenize_element(element):
        if isinstance(element, Doctype):
            return ()
        if isinstance(element, NavigableString):
            return tokenize(element)
        if isinstance(element, Tag):
            ts = ()
            ts += (format_tag(element),)
            for child in element.children:
                ts += tokenize_element(child)
            ts += (format_endtag(element),)
            return ts
        raise ValueError('Unexpected type: {}'.format(type(element)))

    soup = BeautifulSoup(doc.html, 'lxml')
    tokens = tokenize_element(soup)
    return doc.set_tokens(tokens)


def html_tokenize(docs: DocsIterator, use_attributes=False) -> DocsIterator:
    log.info('HTML tokenizing documents')
    pool = Pool(os.cpu_count())
    tokenize = partial(_html_tokenize, use_attributes=use_attributes)
    return pool.map(tokenize, docs)


def lowercase(docs: DocsIterator) -> DocsIterator:
    log.info('Lowercase documents')
    for doc in docs:
        lowercase_tokens = tuple(t.lower() for t in doc.tokens)
        yield doc.set_tokens(lowercase_tokens)


def maxlen(docs: DocsIterator):
    return max(len(doc.html) for doc in docs)
