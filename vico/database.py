import pickle

from serum import Component, immutable, inject
from pony.orm import Database, Optional, PrimaryKey, db_session

from vico.console_arguments import ConsoleArguments
from vico.html_document import HTMLDocument
import os


_db = Database()


class DocumentDatabase(Component):
    args = inject(ConsoleArguments)

    class Document(_db.Entity):
        id = PrimaryKey(int, auto=True)
        html = Optional(str)
        brand = Optional(str)
        gtin13 = Optional(str)
        ean = Optional(str)
        asin = Optional(str)
        sku = Optional(str)
        price = Optional(float)
        currency = Optional(str)
        vendor = Optional(str)
        language = Optional(str)
        tokens = Optional(bytes)
        brand_bio_labels = Optional(bytes)
        windows_5 = Optional(bytes)
        windows_11 = Optional(bytes)
        windows_21 = Optional(bytes)

    @db_session
    def save_documents(self, documents: [HTMLDocument]):
        for doc in documents:
            db_doc = self.Document(
                html=doc.html,
                brand=doc.brand,
                vendor=doc.vendor,
                language=doc.language,
                tokens=pickle.dumps(doc.tokens),
                brand_bio_labels=pickle.dumps(doc.brand_bio_labels),
                windows_5=pickle.dumps(doc.windows_5),
                windows_11=pickle.dumps(doc.windows_11),
                windows_21=pickle.dumps(doc.windows_21)
            )
            if doc.gtin13 is not None:
                db_doc.gtin13 = str(doc.gtin13)
            if doc.ean is not None:
                db_doc.ean = str(doc.ean)
            if doc.asin is not None:
                db_doc.asin = str(doc.asin)
            if doc.sku is not None:
                db_doc.sku = str(doc.sku)
            if doc.price is not None:
                try:
                    db_doc.price = float(doc.price.replace(',', '.'))
                except:
                    pass
            if doc.currency is not None:
                db_doc.currency = doc.currency
            _db.commit()

    @db_session
    def load_documents(self) -> [HTMLDocument]:
        docs = []
        n_samples = self.args.get().n_samples
        n_samples = n_samples if n_samples else self.Document.select().count()
        for db_doc in self.Document.select().limit(n_samples):
            doc = HTMLDocument(
                html=db_doc.html,
                brand=db_doc.brand,
                gtin13=db_doc.gtin13,
                ean=db_doc.ean,
                asin=db_doc.asin,
                sku=db_doc.asin,
                price=db_doc.price,
                currency=db_doc.currency,
                vendor=db_doc.vendor,
                language=db_doc.language,
                tokens=pickle.loads(db_doc.tokens),
                brand_bio_labels=pickle.loads(db_doc.brand_bio_labels),
                windows_5=pickle.loads(db_doc.windows_5),
                windows_11=pickle.loads(db_doc.windows_11),
                windows_21=pickle.loads(db_doc.windows_21)
            )
            docs.append(doc)
        return docs

    def __init__(self):
        path = self.args.get().database_path
        _db.bind(provider='sqlite', filename=path, create_db=False)
        _db.generate_mapping(create_tables=True)
