import gzip
import os
from urllib.parse import urlparse

import scrapy
from itertools import islice


class AmazonDownloadSpider(scrapy.Spider):
    name = 'AmazonDownloadSpider'
    custom_settings = {
        'USER_AGENT': "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36"
    }

    def __init__(self,
                 labels_file='metadata.json.gz',
                 output_folder='data',
                 **kwargs):
        super().__init__(**kwargs)
        self.start_urls = self._read_start_urls(labels_file)
        self.output_folder = output_folder

    def parse(self, response):
        self.save(response)
        return []

    def save(self, response):
        language = self.get_language(response.url)
        asin = self.get_asin(response.url)
        output_path = os.path.join(self.output_folder, language)
        os.makedirs(output_path, exist_ok=True)
        filename = f'{asin}.html'
        filepath = os.path.join(output_path, filename)
        with open(filepath, 'w') as f:
            f.write(response.body.decode())

    def _parse_labels(self, path):
        print('Reading from', path)
        with gzip.open(path, 'r') as g:
            for l in g:
                yield eval(l)

    def create_urls(self, asin):
        base = 'https://amazon'
        domains = ['.com', '.de', '.fr', '.es', '.it']
        page = f'/dp/{asin}'
        for domain in domains:
            yield f'{base}{domain}{page}'

    def _read_start_urls(self, labels_file):
        all_urls = []
        for line in self._parse_labels(labels_file):
            urls = self.create_urls(line['asin'])
            all_urls.extend(list(urls))
        return all_urls


    def get_language(self, url):
        domain = urlparse(url).hostname.split('.').pop()
        return domain if domain != 'com' else 'en'


    def get_asin(self, url):
        return urlparse(url).path.split('/').pop()