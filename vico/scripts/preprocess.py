import argparse

from serum import Environment

from vico import Config
from vico.argument_provider import ArgumentProvider
from vico.database import DocumentDatabase
from vico.preprocess import remove_useless_tags, html_tokenize, lowercase


def parse_arguments():
    parser = argparse.ArgumentParser(
        'Perform preprocessing of documents '
        'and create windows for structured prediction')
    parser.add_argument('--database-path', help='path to sqlite database', default='../data/all_documents.sqlite')
    return parser.parse_args()


def run(database_path):
    config = Config(database_path=database_path)
    ArgumentProvider.config = config
    with Environment(ArgumentProvider):
        db = DocumentDatabase()
        documents = db.load_documents()
        cleaned_documents = remove_useless_tags(documents)
        tokenized_documents = html_tokenize(cleaned_documents)
        lowercase_documents = lowercase(tokenized_documents)
        db.save_documents(lowercase_documents)


if __name__ == '__main__':
    run(**vars(parse_arguments()))

