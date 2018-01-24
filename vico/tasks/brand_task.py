from serum import immutable
from sklearn.metrics import f1_score

from vico.html_document import HTMLDocument
from vico.tasks.classification_task import ClassificationTask


def format_brand(brand: str) -> str:
    return brand.lower()


class BrandTask(ClassificationTask):
    target = immutable(True)

    @property
    def scoring_function(self):
        return 'f1'

    def __init__(self):
        super().__init__()
        self.best_score = float('-inf')

    def is_best_score(self, f1):
        return f1 > self.best_score

    def label(self, document: HTMLDocument) -> str:
        return format_brand(document.brand)

    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        return [d for d in documents
                if d.brand is not None and isinstance(d.brand, str)]

    @property
    def name(self):
        return 'brand'

    def score(self, documents: [HTMLDocument]):
        documents = self.filter_documents(documents)
        sequences, labels = self.vocabulary.make_batch(
            documents,
            self.encode_labels
        )
        predictions = self._model.predict(sequences)
        predicted_int_labels = predictions.argmax(axis=1)
        actual_labels = [self.label(t) for t in documents]
        actual_int_labels = self.label_encoder.transform(actual_labels)
        return f1_score(actual_int_labels, predicted_int_labels, average='macro')
