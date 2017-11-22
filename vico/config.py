import logging
from datetime import datetime
from typing import Dict, Any
from f import Immutable
import time


def now() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


class Config(Immutable):
    log_level: int = logging.INFO
    data_dir: str = "./data"
    folds: int = 5
    embedding_dim: int = 50
    patience: int = 5
    output_file: str = now() + ".csv"
    output_dir: str = "./reports"
    use_attributes: bool = False
    epochs: int = 500
    filters = 10
    filter_sizes = [2, 3, 4, 5]
    seed = int(time.time())

    def __repr__(self) -> str:
        return """Config(log_level={},
     data_dir={},
     folds={},
     patience={},
     embedding_dim={},
     filters={},
     output_dir={},
     output_file={})""".format(self.log_level,
                               self.data_dir,
                               self.folds,
                               self.patience,
                               self.embedding_dim,
                               self.filters,
                               self.output_dir,
                               self.output_file)

    def hyper_parameters(self) -> Dict[str, Any]:
        return {
            'embedding_dim': self.embedding_dim,
            'filters': self.filters
        }
