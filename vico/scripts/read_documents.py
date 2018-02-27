import argparse

import os
import re
from urllib.parse import urlparse

import langdetect
import math
import pandas
from bs4 import BeautifulSoup
from serum import Environment

from vico import Config
from vico.argument_provider import ArgumentProvider
from vico.database import DocumentDatabase
from vico.html_document import HTMLDocument
from vico.scripts.download import format_file_name


def parse_args():
    parser = argparse.ArgumentParser('Read HTML pages into an sqlite database')
    parser.add_argument('label_path', help='path to wide format .csv with document labels')
    parser.add_argument('--page-folder', default='data/pages')
    parser.add_argument('--database-path', default='data/all_docs.sqlite')
    return parser.parse_args()


def vendor(url):
    parse = urlparse(url)
    pattern = '(www\.)?([a-z\-]*)(\.[a-z]*)'
    m = re.match(pattern, parse.hostname)
    return m[2] if m else None


def language(html):
    soup = BeautifulSoup(html, 'lxml')
    text = soup.get_text()
    return langdetect.detect(text) if text else None


def read_documents(label_path, page_folder, database_path):
    def get(value, convert=str):
        return convert(value) if not math.isnan(value) else None
    data = pandas.read_csv(label_path)
    documents = []
    for _, row in data.iterrows():
        filename = format_file_name(row.product_page_url)
        path = os.path.join(page_folder, filename)
        with open(path) as f:
            html = f.read()
            document = HTMLDocument(
                html=html,
                language=language(html),
                vendor=vendor(row.product_page_url),
                brand=get(row.brand),
                ean=get(row.ean, convert=int),
                asin=get(row.asin, convert=int),
                sku=get(row.sku),
                price=get(row.price, convert=float),
                currency=get(row.currency),
                gtin13=get(row.gtin13, convert=int)
            )
            documents.append(document)
        exists = os.path.exists(database_path)
        if not exists:
            os.makedirs(path)
            with open(database_path) as f:
                pass
        config = Config(database_path=database_path)
        ArgumentProvider.config = config
        with Environment(ArgumentProvider):
            db = DocumentDatabase()
            db.save_documents(documents, overwrite=not exists)


if __name__ == '__main__':
    args = parse_args()
    read_documents(**vars(args))
