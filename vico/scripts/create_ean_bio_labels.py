from serum import Environment
from vico.database import DocumentDatabase
import pickle
import re

from vico.html_document import HTMLDocument

with Environment():
    database = DocumentDatabase()
    docs = database.load_documents()
    with open('data/indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    indices2tokens = {v: k for k, v in indices.items()}
    ean_docs = []
    for doc in docs:
        if doc.ean is None:
            ean_docs.append(doc)
            continue
        formatted_ean = re.sub(r'\.[0-9]+', '', doc.ean)
        ean_bio_labels = []
        for window in doc.windows:
            tokens = [indices2tokens[i] if i != 0 else 'PAD' for i in window]
            middle_token = tokens[len(tokens) // 2]
            if middle_token == formatted_ean:
                ean_bio_labels.append(1)
            else:
                ean_bio_labels.append(0)
        assert len(ean_bio_labels) == len(doc.windows)
        doc_dict = doc._asdict()
        doc_dict['ean_bio_labels'] = ean_bio_labels
        ean_docs.append(HTMLDocument(**doc_dict))
    assert len(docs) == len(ean_docs)
    database.save_documents(ean_docs)

