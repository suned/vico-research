from serum import Component, inject

from vico.console_arguments import ConsoleArguments
from vico.tasks import Tasks

import logging
import random
from vico.tasks.task import Task

log = logging.getLogger('vico.train')


class ModelTrainer(Component):
    tasks = inject(Tasks)
    args = inject(ConsoleArguments)

    def fit_tasks(self):
        config = self.args.get()

        def find_best_epoch() -> int:
            log.info('Finding best epoch')
            while True:
                task = random.choice(self.tasks)
                epochs_without_improvement = task.epochs_without_improvement
                patience_exceeded = epochs_without_improvement > config.patience
                if task.target and patience_exceeded:
                    total_epochs = sum([t.epoch - config.patience
                                        for t in self.tasks])
                    log.info(
                        'Patience exceeded on task %s. Found best epoch: %i',
                        task.name,
                        total_epochs
                    )
                    return total_epochs
                task.fit_early_stopping()

        def fit_on_all_data(epochs) -> [Task]:
            log.info('Fitting on all data')
            for epoch in range(epochs):
                log.info('Epoch: %i', epoch + 1)
                task = random.choice(self.tasks)
                task.fit()
            return self.tasks

        best_epoch = find_best_epoch()
        self.tasks.recompile()
        fit_on_all_data(best_epoch)
