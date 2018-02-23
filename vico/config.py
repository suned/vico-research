import logging
from datetime import datetime
from typing import Dict, Any, List
from f import Immutable
import time


def now() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


class Config(Immutable):
    log_level: int = logging.INFO
    patience: int = 5
    output_file: str = now() + ".csv"
    output_dir: str = "./reports"
    filters: int = 10
    filter_sizes: List[int] = [2, 3, 4, 5]
    targets: List[str] = ['brand']
    seed: int = int(time.time())
    embedding_path: str = 'data/transformed_embedding.pkl'
    n_samples: int = None
    window_size: int = 5
    database_path: str = '../data/all_docs.sqlite'
    skip: str = []
    indices_path: str = 'data/indices.pkl'

    def __repr__(self) -> str:
        return """Config(log_level={},
     patience={},
     filters={},
     output_dir={},
     output_file={})""".format(self.log_level,
                               self.patience,
                               self.filters,
                               self.output_dir,
                               self.output_file)

    @property
    def hyper_parameters(self) -> Dict[str, Any]:
        return {
            'filters': self.filters,
            'patience': self.patience,
            'window_size': self.window_size,
        }
