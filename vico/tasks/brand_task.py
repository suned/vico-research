from keras import Model, Input
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from vico.html_document import Tokenization
from vico.tasks.task import Task
from numpy import ndarray, array

from vico.types import Tokenizations


def format_brand(brand: str) -> str:
    return brand.lower()


class BrandTask(Task):
    def label(self, tokenization: Tokenization) -> str:
        return format_brand(tokenization.document.brand)

    def filter_tokenizations(self, tokenizations: Tokenizations) -> Tokenizations:
        return tokenizations.filter(lambda t: t.document.brand is not None)

    label_encoder = None
    one_hot_encoder = None

    @property
    def name(self):
        return 'brand'

    def encode_labels(self, tokenizations: Tokenizations) -> ndarray:
        if self.label_encoder is None:
            raise RuntimeError('Compile model must be called before label')
        brands = tokenizations.map(self.label)
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
        shared_tensor = self._shared_layers(input_layer)
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
