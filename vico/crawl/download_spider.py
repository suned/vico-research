import os
from urllib.parse import urljoin
import scrapy
import pandas
from f import Immutable

from vico import preprocess
from vico.html_document import HTMLDocument


class Link(Immutable):
    url: str
    tag: None


class DownloadSpider(scrapy.Spider):
    custom_settings = {
        'USER_AGENT': "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36"
    }

    def __init__(self,
                 domain,
                 labels_file='data/CosmeticsTestData.csv',
                 output_folder='data/unlabeled',
                 language='de',
                 **kwargs):
        super().__init__(**kwargs)
        output_folder = os.path.join(output_folder, language)
        os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder
        self.language = language
        self.domain = domain
        self.base = 'http://www.{}.{}'.format(domain, language)
        self.start_urls = self._parse_start_urls(labels_file)

    def _parse_start_urls(self, labels_csv):
        labels = pandas.read_csv(labels_csv, sep=';')
        return [url for url in labels.url
                if self.domain in url and self.language in url]

    def parse(self, response):
        self.save(response)
        for link in self.get_links(response):
            if not self.exists(link.url):
                yield response.follow(link.tag, self.parse)

    def get_links(self, response):
        raise NotImplementedError()

    def save(self, response):
        doc = HTMLDocument(html=response.body.decode())
        doc = preprocess.remove_tags(doc)
        output_file = self.format_file_name(response.url)
        with open(output_file, 'w') as f:
            f.write(doc.html)

    def format_file_name(self, url):
        clean_url = url.replace('/', '_')
        output_file = os.path.join(self.output_folder, clean_url + '.html')
        return output_file

    def exists(self, url):
        url = urljoin(self.base, url)
        path = self.format_file_name(url)
        return os.path.isfile(path)
