import logging
from abc import abstractmethod
from typing import Any

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from vico.html_document import HTMLDocument
from vico.tasks.task import Task
from numpy import ndarray
from keras import Model, Input
from keras.layers import Dense


log = logging.getLogger('vico.tasks.classification_task')


class ClassificationTask(Task):
    def __init__(self):
        self.label_encoder = None
        self.one_hot_encoder = None
        super().__init__()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def scoring_function(self) -> str:
        return 'crossentropy'

    @abstractmethod
    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        pass

    @abstractmethod
    def label(self, document: HTMLDocument) -> Any:
        pass

    def encode_labels(self, documents: [HTMLDocument]) -> ndarray:
        if self.label_encoder is None:
            raise RuntimeError('Compile model must be called before label')
        brands = [self.label(d) for d in documents]
        int_labels = self.label_encoder.transform(brands).reshape(-1, 1)
        return self.one_hot_encoder.transform(int_labels)

    def compile_model(self) -> Model:
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        int_labels = self.label_encoder.fit_transform(
            list(self.unique_labels)
        ).reshape(-1, 1)
        self.one_hot_encoder.fit(int_labels)
        input_layer = Input(
            shape=(self.input_length,),
            name='token_input'
        )
        shared_tensor = self.shared_layers(input_layer)
        classes = len(self.unique_labels)
        output = Dense(
            classes,
            activation='softmax',
            name='output_layer'
        )(shared_tensor)
        model = Model(
            inputs=input_layer,
            outputs=output,
        )
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy'
        )
        return model
