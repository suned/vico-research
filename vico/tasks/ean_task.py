from serum import immutable
from sklearn.metrics import f1_score

from vico.html_document import HTMLDocument
from vico.shared_layers import SharedLayers
from vico.tasks.sequence_classification_task import SequenceClassificationTask


class EANTask(SequenceClassificationTask):
    target = immutable(True)

    @property
    def name(self) -> str:
        return 'EAN'

    def label(self, document: HTMLDocument):
        return document.ean_bio_labels

    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        return [d for d in documents
                if d.brand_bio_labels is not None
                and isinstance(d.ean_bio_labels, list)]

    def _score(self, windows, labels) -> float:
        predictions = self._model.predict(windows)
        predictions = ['O' if s < .5 else 'I_EAN' for s in predictions]
        labels = ['O' if l < .5 else 'I_EAN' for l in labels]
        return f1_score(labels, predictions, average='macro')

    @property
    def scoring_function(self):
        return 'f1'

    def __init__(self, shared_layers: SharedLayers, all_data=False):
        super().__init__(shared_layers, all_data)
        self.best_score = float('-inf')

    def is_best_score(self, f1):
        return f1 > self.best_score
