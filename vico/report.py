import os
import pandas
from keras.models import Model
from pymonad import List
import logging
from vico import vocabulary, label, Config

log = logging.getLogger("vico.report")


def save(model: Model,
         train_docs: List,
         test_docs: List,
         config: Config) -> None:
    log.info('Saving validation metrics')
    batcher = vocabulary.batcher(
        train_docs,
        labeller=label.price
    )
    train_sequences, train_labels = batcher(train_docs)
    test_sequences, test_labels = batcher(test_docs)
    train_loss = model.test_on_batch(train_sequences, train_labels)
    test_loss = model.test_on_batch(test_sequences, test_labels)
    data = pandas.DataFrame(
        [config.hyper_parameters()]
    )
    data['train_samples'] = len(train_docs)
    data['test_samples'] = len(test_docs)
    data['train_loss'] = train_loss
    data['test_loss'] = test_loss
    out_dir = config.output_dir
    output_file = config.output_file
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, output_file)
    exists = os.path.isfile(output_path)
    data.to_csv(output_path, index=False, mode='a', header=not exists)
