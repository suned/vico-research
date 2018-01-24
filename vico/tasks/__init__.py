import logging

from serum import inject, Singleton

from vico.console_arguments import ConsoleArguments
from vico.shared_layers import SharedLayers
from vico.tasks.brand_task import BrandTask
from vico.tasks.language_task import LanguageTask
from vico.tasks.price_task import PriceTask
from vico.tasks.task import Task
from vico.tasks.vendor_task import VendorTask


log = logging.getLogger('vico.tasks')


def task_factory(target: str) -> Task:
    return {
        'price': PriceTask(),
        'vendor': VendorTask(),
        'brand': BrandTask(),
        'language': LanguageTask()
    }[target]


class Tasks(Singleton):
    args = inject(ConsoleArguments)
    shared_layers = inject(SharedLayers)

    def __init__(self):
        config = self.args.get()
        self._tasks = [task_factory(target) for target in config.targets]

    def recompile(self):
        log.info('recompiling models')
        self.shared_layers.recompile()
        [t.recompile() for t in self]

    def __iter__(self):
        return iter(self._tasks)

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, item):
        return self._tasks[item]