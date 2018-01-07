import os
import pandas
from f import List
import logging
from vico import Config
from vico.tasks.task import Task

log = logging.getLogger("vico.report")


def save(fitted_tasks: [Task],
         train_docs: List,
         test_docs: List,
         config: Config) -> None:
    log.info('Saving validation metrics')
    data = pandas.DataFrame(
        [config.hyper_parameters()]
    )
    for task in fitted_tasks:
        train_loss = task.test_loss(train_docs)
        test_loss = task.test_loss(test_docs)

        data[task.name + '_train_samples'] = len(
            task.filter_tokenizations(train_docs)
        )
        data[task.name + '_test_samples'] = len(
            task.filter_tokenizations(test_docs)
        )
        data[task.name + '_train_loss'] = train_loss
        data[task.name + '_test_loss'] = test_loss
    out_dir = config.output_dir
    output_file = config.output_file
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, output_file)
    exists = os.path.isfile(output_path)
    data.to_csv(output_path, index=False, mode='a', header=not exists)
