import logging

from serum import inject, Component

from vico.console_arguments import ConsoleArguments
from vico.shared_layers import SharedLayersBuilder
from vico.tasks.brand_task import BrandTask
from vico.tasks.ean_task import EANTask
from vico.tasks.language_task import LanguageTask
from vico.tasks.price_task import PriceTask
from vico.tasks.task import Task
from vico.tasks.vendor_task import VendorTask


log = logging.getLogger('vico.tasks')


class TaskBuilder(Component):
    args = inject(ConsoleArguments)
    shared_layers_builder = inject(SharedLayersBuilder)

    def build(self, all_data=False) -> [Task]:
        shared_layers = self.shared_layers_builder.build()

        def task_factory(target: str) -> Task:
            return {
                'price': PriceTask(shared_layers, all_data=all_data),
                'vendor': VendorTask(shared_layers, all_data=all_data),
                'brand': BrandTask(shared_layers, all_data=all_data),
                'language': LanguageTask(shared_layers, all_data=all_data),
                'ean': EANTask(shared_layers, all_data=all_data)
            }[target]
        config = self.args.get()
        log.info('recompiling models')

        return [task_factory(target) for target in config.targets]