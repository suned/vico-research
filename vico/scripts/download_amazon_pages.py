import argparse

import requests
from concurrent import futures
import os
import gzip
import logging


log = logging.getLogger('vico.download_amazon_pages')
log.setLevel(logging.INFO)


def parse(path):
    print('Reading from', path)
    with gzip.open(path, 'r') as g:
        for l in g:
            yield eval(l)


def create_urls(asin):
    base = 'https://amazon'
    domains = ['.com', '.de', '.fr', '.es', '.it']
    page = f'/dp/{asin}'
    for domain in domains:
        yield domain, f'{base}{domain}{page}'


def download(url, output_path):
    headers = {
        'USER_AGENT': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:58.0) Gecko/20100101 Firefox/58.0"
    }
    print('making request to', url)
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print('Downloading %s' % url)
        path, _ = os.path.split(output_path)
        os.makedirs(path, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(response.text)
    else:
        print('Bad status code %i for url %s' % (response.status_code, url))


def create_output_path(asin, folder, domain):
    filename = f'{asin}.html'
    return os.path.join(folder, domain, filename)


def process(json, folder):
    asin = json['asin']
    urls = create_urls(asin)
    for domain, url in urls:
        output_path = create_output_path(asin, folder, domain)
        download(url, output_path)


def main(folder, gzip_path):
    with futures.ThreadPoolExecutor(2) as executor:
        for json in parse(gzip_path):
            executor.submit(process, json, folder)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('gzip_path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
