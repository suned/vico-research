import re
from urllib.parse import urlparse
import langdetect
import math

from bs4 import BeautifulSoup
import pandas


def vendor(url):
    parse = urlparse(url)
    pattern = '(www\.)?([a-z\-]*)(\.[a-z]*)'
    m = re.match(pattern, parse.hostname)
    return m[2] if m else None


def language(path):
    if type(path) == float and math.isnan(path):
        return None
    with open(path) as f:
        data = f.read()
    soup = BeautifulSoup(data, 'lxml')
    text = soup.get_text()
    return langdetect.detect(text) if text else None


labels = pandas.read_csv('data/labels.csv')
vendors = [vendor(row.url) for _, row in labels.iterrows()]
languages = [language(row.path) for _, row in labels.iterrows()]
import ipdb
ipdb.sset_trace()
labels['vendor'] = vendors
labels['language'] = languages
labels.to_csv('data/labels.csv', index=False)
