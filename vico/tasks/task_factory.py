from vico.tasks.brand_task import BrandTask
from vico.tasks.language_task import LanguageTask
from vico.tasks.vendor_task import VendorTask
from vico.tasks.price_task import PriceTask
from vico.tasks.task import Task


def create_factory(
        train_tokenizations,
        test_tokenizations,
        vocabulary, 
        shared_layers):
    def get(target: str) -> Task:
        return {
            'price': PriceTask(
                train_tokenizations,
                test_tokenizations,
                vocabulary, 
                shared_layers
            ),
            'vendor': VendorTask(
                train_tokenizations,
                test_tokenizations,
                vocabulary, 
                shared_layers
            ),
            'brand': BrandTask(
                train_tokenizations,
                test_tokenizations,
                vocabulary,
                shared_layers
            ),
            'language': LanguageTask(
                train_tokenizations,
                test_tokenizations,
                vocabulary,
                shared_layers
            )
        }[target]
    return get