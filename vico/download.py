import os
import pandas
import requests
import argparse


def format_file_name(url):
    clean_url = (url
                 .replace('http://', '')
                 .replace('https://', '')
                 .replace('www.', 'www_')
                 .replace('.apsx', 'aspx')
                 .replace('/', '_')
                 .replace('.', '_'))
    return clean_url if clean_url.endswith('.html') else clean_url + '.html'


def download_all(label_path, output_folder):
    label_frame = pandas.read_csv(label_path)
    for url in label_frame.product_page_url:
        download(url, output_folder)


def download(url, output_folder):
    file_name = format_file_name(url)
    file_path = os.path.join(output_folder, file_name)
    response = requests.get(url)
    if response.status_code != 200:
        return
    with open(file_path, 'w') as f:
        f.write(response.text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_path')
    parser.add_argument('--output-folder', default='data/pages')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    download_all(**vars(args))
