from serum import Component, inject

from vico.console_arguments import ConsoleArguments
from vico.tasks import TaskBuilder, Task

import logging
import random

log = logging.getLogger('vico.train')


class ModelTrainer(Component):
    task_builder = inject(TaskBuilder)
    args = inject(ConsoleArguments)

    def fit_tasks(self, all_data=False) -> [Task]:
        config = self.args.get()

        def find_best_epoch(ts) -> int:
            log.info('Finding best epoch')
            while True:
                task = random.choice(ts)
                epochs_without_improvement = task.epochs_without_improvement
                patience_exceeded = epochs_without_improvement > config.patience
                if task.target and patience_exceeded:
                    total_epochs = (task.epoch - config.patience) * len(ts)
                    log.info(
                        'Patience exceeded on task %s. Found best epoch: %i',
                        task.name,
                        total_epochs
                    )
                    return total_epochs
                task.fit_early_stopping()

        def fit_on_all_data(epochs, ts):
            log.info('Fitting on all data')
            for epoch in range(epochs):
                log.info('Epoch: %i of %i', epoch + 1, epochs)
                task = random.choice(ts)
                task.fit()
        tasks = self.task_builder.build(all_data)
        best_epoch = find_best_epoch(tasks)
        tasks = self.task_builder.build(all_data)
        fit_on_all_data(best_epoch, tasks)
        return tasks
