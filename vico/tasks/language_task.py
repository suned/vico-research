from vico.html_document import HTMLDocument
from vico.tasks.classification_task import ClassificationTask


class LanguageTask(ClassificationTask):
    def label(self, document: HTMLDocument) -> str:
        return document.language

    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        return [d for d in documents
                if d.language is not None and isinstance(d.language, str)]

    @property
    def name(self):
        return 'language'
