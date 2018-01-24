from keras import Model, Input
from keras.layers import Dense

from vico.html_document import HTMLDocument
from vico.tasks.task import Task
from numpy import ndarray, array


class PriceTask(Task):
    @property
    def scoring_function(self) -> str:
        return 'mse'

    def filter_documents(self, documents: [HTMLDocument]) -> [HTMLDocument]:
        return [d for d in documents if d.price is not None]

    @property
    def name(self):
        return 'price'

    def label(self, document: HTMLDocument) -> float:
        return document.price

    def encode_labels(self, documents: [HTMLDocument]) -> ndarray:
        return array([self.label(d) for d in documents])

    def compile_model(self) -> Model:
        input_layer = Input(
            shape=(self.input_length,),
            name='token_input'
        )
        shared_tensor = self.shared_layers(input_layer)
        output = Dense(
            1,
            activation='linear',
            name='output_layer'
        )(shared_tensor)
        model = Model(
            inputs=input_layer,
            outputs=output,
        )
        model.compile(
            optimizer='adam',
            loss='mse'
        )
        return model
