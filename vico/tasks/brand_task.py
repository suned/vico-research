from serum import immutable
from sklearn.metrics import f1_score
from numpy import ndarray
from vico.html_document import HTMLDocument
from vico.shared_layers import SharedLayers
from .sequence_classification_task import SequenceClassificationTask


def format_brand(brand: str) -> str:
    return brand.lower()


class BrandTask(SequenceClassificationTask):

    def label(self, document):
        return document.brand_bio_labels

    target = immutable(True)

    @property
    def scoring_function(self):
        return 'f1'

    def __init__(self, shared_layers: SharedLayers):
        super().__init__(shared_layers)
        self.best_score = float('-inf')

    def is_best_score(self, f1):
        return f1 > self.best_score

    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        return [d for d in documents
                if d.brand_bio_labels is not None
                and isinstance(d.brand_bio_labels, list)]

    @property
    def name(self):
        return 'brand'

    def _score(self, windows, labels) -> float:
        predictions = self._model.predict(windows)
        predictions = ['O' if s < .5 else 'I_BRAND' for s in predictions]
        labels = ['O' if l < .5 else 'I_BRAND' for l in labels]
        return f1_score(labels, predictions, average='macro')
