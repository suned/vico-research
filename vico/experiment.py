import logging

from serum import inject, Environment

from vico.console_arguments import ConsoleArguments
from vico.cross_validation_split import CrossValidationSplit, LeaveOneLanguageOut
from vico.evaluator import Evaluator
from vico.model_trainer import ModelTrainer
from vico.tasks import Tasks
from . import configure_root_logger


log = logging.getLogger('vico.experiment')


class Experiment:
    cross_validation_split = inject(CrossValidationSplit)
    model_trainer = inject(ModelTrainer)
    evaluator = inject(Evaluator)
    tasks = inject(Tasks)

    def run(self):
        for i, _ in enumerate(self.cross_validation_split):
            log.info(
                'starting fold %i of %i',
                i + 1,
                len(self.cross_validation_split) + 1
            )
            self.model_trainer.fit_tasks()
            self.evaluator.evaluate()
            self.tasks.recompile()


if __name__ == '__main__':
    with Environment(LeaveOneLanguageOut):
        args = ConsoleArguments().get()
        configure_root_logger(args)
        Experiment().run()

