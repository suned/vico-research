import numpy
from keras import Model, Input
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from vico.html_document import HTMLDocument
from vico.shared_layers import SharedLayers
from vico.tasks.sequence_classification_task import SequenceClassificationTask


class VendorTask(SequenceClassificationTask):
    def __init__(self, shared_layers: SharedLayers):
        super().__init__(shared_layers)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(self.unique_labels))

    def label(self, document: HTMLDocument) -> [str]:
        return [document.vendor for _ in document.windows]

    def stack(self, documents):
        windows = [d.windows for d in documents]
        labels = [self.label(d) for d in documents]
        labels = [self.label_encoder.transform(l) for l in labels]
        labels = [to_categorical(l, num_classes=len(self.unique_labels)) for l in labels]
        windows = numpy.vstack(windows)
        labels = numpy.concatenate(labels)
        return windows, labels

    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        return [d for d in documents if d.vendor is not None
                and isinstance(d.vendor, str)]

    @property
    def name(self):
        return 'vendor'

    def compile_model(self) -> Model:
        window_size = self.config.get().window_size
        input_layer = Input(
            shape=(window_size,)
        )
        shared_tensor = self._shared_layers(input_layer)
        output = Dense(
            len(self.unique_labels),
            activation='softmax'
        )(shared_tensor)
        model = Model(
            inputs=input_layer,
            outputs=output
        )
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy'
        )
        return model
