from keras import Model, Input
from keras.layers import Dense

from vico.html_document import Tokenization
from vico.tasks.task import Task
from numpy import ndarray, array

from vico.types import Tokenizations


class PriceTask(Task):
    def filter_tokenizations(self, tokenizations: Tokenizations) -> Tokenizations:
        return tokenizations.filter(lambda t: t.document.price is not None)

    @property
    def name(self):
        return 'price'

    def label(self, tokenization: Tokenization) -> float:
        return tokenization.document.price

    def encode_labels(self, tokenizations: Tokenizations) -> ndarray:
        return array(tokenizations | self.label)

    def compile_model(self) -> Model:
        input_layer = Input(
            shape=(self.input_length,),
            name='token_input'
        )
        shared_tensor = self._shared_layers(input_layer)
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
