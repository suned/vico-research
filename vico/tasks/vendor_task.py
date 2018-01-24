from vico.html_document import HTMLDocument
from vico.tasks.classification_task import ClassificationTask


class VendorTask(ClassificationTask):
    def label(self, document: HTMLDocument) -> str:
        return document.vendor

    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        return [d for d in documents if d.vendor is not None
                and isinstance(d.vendor, str)]

    @property
    def name(self):
        return 'vendor'
