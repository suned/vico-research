import logging

import os
import pandas
from serum import Component, inject

from vico.console_arguments import ConsoleArguments
from vico.cross_validation_split import CrossValidationSplit
from vico.tasks import Task

log = logging.getLogger("vico.report")


class Evaluator(Component):
    args = inject(ConsoleArguments)
    cross_validation_split = inject(CrossValidationSplit)

    def evaluate(self, tasks: [Task]):
        log.info('Saving validation metrics')
        config = self.args.get()
        data = pandas.DataFrame(
            [config.hyper_parameters]
        )
        for task in tasks:
            data[task.name + '_train_samples'] = len(
                task.filter_documents(self.cross_validation_split.train_documents)
            )
            data[task.name + '_test_samples'] = len(
                task.filter_documents(
                    self.cross_validation_split.test_documents)
            )
            label = '{}_{}_{}'
            test_label = label.format(
                task.name, 'test', task.scoring_function
            )
            train_label = label.format(
                task.name, 'train', task.scoring_function
            )
            data[train_label] = task.train_score()
            data[test_label] = task.test_score()
        data['test_language'] = (self
                                 .cross_validation_split
                                 .test_documents[0]
                                 .language)
        out_dir = config.output_dir
        output_file = config.output_file
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, output_file)
        exists = os.path.isfile(output_path)
        data.to_csv(output_path, index=False, mode='a', header=not exists)
