import pandas
import io
import os
import argparse


def read_as_utf_8(path):
    with open(path, 'rb') as f:
        b = f.read()
    # fix windows line endings
    b = b.replace(b'\r', b'')
    try:
        s = b.decode()
    except UnicodeDecodeError:
        s = b.decode('cp1252')
    return pandas.read_csv(io.StringIO(s), sep=';')


def pivot(dataframe):
    return dataframe.pivot(
        index='product_page_url',
        columns='feature_name',
        values='feature_value'
    ).reset_index()


def output_path(path):
    root, name = os.path.split(path)
    suffix = '_clean'
    name, *_ = name.split('.')
    name = name + suffix + '.csv'
    return os.path.join(root, name)


def run(path):
    dataframe = read_as_utf_8(path)
    dataframe = dataframe.drop_duplicates()
    normalized = pivot(dataframe)
    normalized.to_csv(output_path(path), index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))
