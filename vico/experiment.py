import logging

from serum import inject, Environment

from vico.console_arguments import ConsoleArguments
from vico.cross_validation_split import CrossValidationSplit, LeaveOneLanguageOut
from vico.evaluator import Evaluator
from vico.model_trainer import ModelTrainer
from . import configure_root_logger


log = logging.getLogger('vico.experiment')


class Experiment:
    args = inject(ConsoleArguments)
    cross_validation_split = inject(CrossValidationSplit)
    model_trainer = inject(ModelTrainer)
    evaluator = inject(Evaluator)

    def run(self):
        if not self.args.get().skip_validation:
            for i, _ in enumerate(self.cross_validation_split):
                log.info(
                    'starting fold %i of %i',
                    i + 1,
                    len(self.cross_validation_split) + 2
                )
                tasks = self.model_trainer.fit_tasks()
                self.evaluator.evaluate(tasks)
        log.info('Fitting on all data')
        tasks = self.model_trainer.fit_tasks(all_data=True)
        for task in tasks:
            task.save()


if __name__ == '__main__':
    with Environment(LeaveOneLanguageOut):
        args = ConsoleArguments().get()
        configure_root_logger(args)
        Experiment().run()

