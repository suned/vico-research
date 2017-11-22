import os
import pandas
from keras.models import Model
from f import List
import logging
from vico import label, Config
from vico.vocabulary import Vocabulary

log = logging.getLogger("vico.report")


def save(model: Model,
         vocabulary: Vocabulary,
         train_docs: List,
         test_docs: List,
         config: Config) -> None:
    log.info('Saving validation metrics')
    train_sequences, train_labels = vocabulary.make_batch(
        train_docs,
        labeller=label.price
    )
    test_sequences, test_labels = vocabulary.make_batch(
        test_docs,
        labeller=label.price
    )
    train_loss = model.evaluate(train_sequences, train_labels)
    test_loss = model.evaluate(test_sequences, test_labels)
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
