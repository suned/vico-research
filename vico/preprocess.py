import logging
from functools import partial
from multiprocessing.pool import Pool
import os
from pymonad import curry, List

import nltk
from bs4 import BeautifulSoup, Tag, Doctype, NavigableString

from vico.html_document import HTMLDocument, Tokens
from vico.types import Docs

log = logging.getLogger('vico.preprocess')


def _useless_tags() -> List:
    return List(*[
        'script',
        'style',
        'button',
        'input',
        'meta',
        'noscript',
        'link'
    ])


def _remove_tags(doc: HTMLDocument) -> HTMLDocument:
    nothing = ''
    soup = BeautifulSoup(doc.html, 'lxml')
    for tag in _useless_tags():
        for element in soup.find_all(tag):
            element.replace_with(nothing)
    html = str(soup)
    return doc.set_html(html)


@curry
def remove_useless_tags(docs: Docs) -> Docs:
    log.info('Removing tags: %s', ', '.join(_useless_tags()))
    pool = Pool(os.cpu_count())
    pdocs = pool.map(_remove_tags, docs)
    return Docs(*pdocs)


def _html_tokenize(use_attributes, doc: HTMLDocument) -> HTMLDocument:
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
    return doc.set_tokens(Tokens(*tokens))


@curry
def html_tokenize(use_attributes, docs: Docs) -> Docs:
    log.info('HTML tokenizing documents')
    pool = Pool(os.cpu_count())
    tokenize = partial(_html_tokenize, use_attributes)
    pdocs = pool.map(tokenize, docs)
    return Docs(*pdocs)


@curry
def simple_tokenize(docs: Docs) -> Docs:
    def tokenize(doc: HTMLDocument) -> HTMLDocument:
        tokens = doc.html.split(' ')
        return doc.set_tokens(Tokens(*tokens))
    return tokenize * docs


@curry
def lowercase(docs: Docs) -> Docs:
    def to_lower(doc: HTMLDocument) -> HTMLDocument:
        lower_tokens = (lambda t: t.lower()) * doc.tokens
        return doc.set_tokens(lower_tokens)

    log.info('Lowercase documents')
    return to_lower * docs


def maxlen(docs: Docs) -> int:
    return max(len(doc.html) for doc in docs)
