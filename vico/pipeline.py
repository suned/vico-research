import nltk
import langdetect
import keras
import pickle
import numpy
from bs4 import BeautifulSoup, Tag, Doctype, NavigableString, Comment

from typing import *


def html_tokenize_document(html: str, language: str) -> Tuple[str]:
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
        return tuple(nltk.word_tokenize(text, language=language))

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

    soup = BeautifulSoup(html, 'lxml')
    return tokenize_element(soup)


def _useless_tags() -> Tuple[str, ...]:
    return (
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


def window(iterable, left, right, padding=None, step=1):
    """Make a sliding window iterator with padding.

    Iterate over `iterable` with a step size `step` producing a tuple for each element:
        ( ... left items, item, right_items ... )
    such that item visits all elements of `iterable` `step`-steps aside, the length of
    the left_items and right_items is `left` and `right` respectively, and any missing
    elements at the start and the end of the iteration are padded with the `padding`
    item.
    For example:

    >>> list( window( range(5), 1, 2 ) )
    [(None, 0, 1, 2), (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, None), (3, 4, None, None)]
    """
    from itertools import islice, repeat, chain
    from collections import deque

    n = left + right + 1

    iterator = chain(iterable, repeat(padding, right))

    elements = deque(repeat(padding, left), n)
    elements.extend(islice(iterator, right - step + 1))

    while True:
        for i in range(step):
            elements.append(next(iterator))
        yield tuple(elements)


def remove_tags(html: str) -> str:
    nothing = ''
    soup = BeautifulSoup(html, 'lxml')
    for tag in _useless_tags():
        for element in soup.find_all(tag):
            element.replace_with(nothing)
    return str(soup)


def lowercase(html: str) -> str:
    return html.lower()


class EANPipeline:
    model: keras.Model
    indices: Dict[str, int]
    padding_index = 0

    @property
    def out_of_vocab_index(self):
        return len(self.indices) + 1

    def tokenize(self, html: str, language: str) -> Tuple[str]:
        return html_tokenize_document(html, language)

    def preprocess(self, html):
        without_useless_tags = remove_tags(html)
        return lowercase(without_useless_tags)

    def tokens2indices(self, tokens: Tuple[str]) -> Tuple[int]:
        return tuple(
            self.indices.get(token, self.out_of_vocab_index) for token in tokens
        )

    def create_windows(self, indices):
        return tuple(
            window(
                indices,
                left=2,
                right=2,
                padding=self.padding_index
            )
        )

    def __call__(self, html: str, language='german') -> List[str]:
        preprocessed_html = self.preprocess(html)
        tokens = self.tokenize(preprocessed_html, language)
        indices = self.tokens2indices(tokens)
        windows = self.create_windows(indices)
        probabilities = self.model.predict(numpy.array(windows))
        return [token for token, probability in zip(tokens, probabilities)
                if probability > .5]

    @staticmethod
    def load(hdf5_path, indices_path) -> 'EANPipeline':
        pipeline = EANPipeline()
        pipeline.model = keras.models.load_model(hdf5_path)
        with open(indices_path, 'rb') as f:
            pipeline.indices = pickle.load(f)
        return pipeline
