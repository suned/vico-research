import ast
import logging
from serum import Component, inject
from vico.database import DocumentDatabase
from vico.html_document import HTMLDocument

log = logging.getLogger('vico.read')


def parse_list(l):
    return ast.literal_eval(l) if l else None


class DataSetReader(Component):
    database = inject(DocumentDatabase)

    def read_documents(self) -> [HTMLDocument]:
        documents = self.database.load_documents()
        return [d for d in documents if len(d.windows) > 0]
