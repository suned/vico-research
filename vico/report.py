import os
import pandas
from keras.models import Model

from vico import vocabulary, args
from vico.types import Docs


def save(model: Model, train_docs: Docs, test_docs: Docs):
    batch_generator = vocabulary.batch_generator(train_docs)
    train_sequences, train_labels = batch_generator(train_docs)
    test_sequences, test_labels = batch_generator(test_docs)
    train_loss = model.test_on_batch(train_sequences, train_labels)
    test_loss = model.test_on_batch(test_sequences, test_labels)
    data = pandas.DataFrame(
        [args.hyper_parameters()]
    )
    data['train_samples'] = len(train_docs)
    data['test_samples'] = len(test_docs)
    data['train_loss'] = train_loss
    data['test_loss'] = test_loss
    out_dir = args.get().output_dir
    output_file = args.get().output_file
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, output_file)
    exists = os.path.isfile(output_path)
    data.to_csv(output_path, index=False, mode='a', header=not exists)
