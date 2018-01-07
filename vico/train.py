import logging
import random

from vico import Config
from vico.shared_layers import SharedLayers
from vico.tasks.task import Task
from vico.tasks.task_factory import create_factory
from vico.types import Tokenizations
from vico.vocabulary import Vocabulary

log = logging.getLogger('vico.train')


def early_stopping(train_tokenizations: Tokenizations,
                   test_tokenizations: Tokenizations,
                   vocabulary: Vocabulary,
                   config: Config) -> [Task]:
    def init_tasks() -> [Task]:
        log.info('Initialising tasks')
        shared_layers = SharedLayers(vocabulary, config)
        factory = create_factory(
            train_tokenizations,
            test_tokenizations,
            vocabulary,
            shared_layers
        )
        return [factory(target) for target in config.targets]

    def find_best_epoch() -> int:
        log.info('Finding best epoch')
        tasks = init_tasks()
        while True:
            task = random.choice(tasks)
            if task.epochs_without_improvement > config.patience:
                total_epochs = sum([t.epoch for t in tasks])
                log.info(
                    'Patience exceeded on task %s. Found best epoch: %i',
                    task.name,
                    total_epochs
                )
                return total_epochs
            task.fit_early_stopping()

    def fit_on_all_data(epochs) -> [Task]:
        log.info('Fitting on all data')
        tasks = init_tasks()
        for epoch in range(epochs):
            task = random.choice(tasks)
            task.fit()
        return tasks

    best_epoch = find_best_epoch()
    return fit_on_all_data(best_epoch)
