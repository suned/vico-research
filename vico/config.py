import logging
from datetime import datetime
from typing import Dict, Any

_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


class Config:
    def __init__(self,
                 bilstm_dim: int = 50,
                 log_level: str = "DEBUG",
                 data_dir: str = "./data",
                 folds: int = 2,
                 embedding_dim: int = 50,
                 output_file: str =_now + ".csv",
                 output_dir: str = "./reports",
                 use_attributes: bool = False,
                 epochs: int = 1) -> None:
        self._bilstm_dim = bilstm_dim
        self._log_level = log_level
        self._data_dir = data_dir
        self._folds = folds
        self._embedding_dim = embedding_dim
        self._output_dir = output_dir
        self._output_file = output_file
        self._use_attributes = use_attributes
        self._epochs = epochs

    @property
    def epochs(self) -> int:
        return self._epochs

    @property
    def use_attributes(self) -> bool:
        return self._use_attributes

    @property
    def bilstm_dim(self) -> int:
        return self._bilstm_dim

    @property
    def log_level(self) -> int:
        return getattr(logging, self._log_level)

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def folds(self) -> int:
        return self._folds

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def output_file(self) -> str:
        return self._output_file

    def __repr__(self) -> str:
        return """Args(bilstm_dim={},
     log_level={},
     data_dir={},
     folds={},
     embedding_dim={},
     output_dir={},
     output_file={})""".format(self.bilstm_dim,
                               self._log_level,
                               self.data_dir,
                               self.folds,
                               self.embedding_dim,
                               self.output_dir,
                               self._output_file)

    def hyper_parameters(self) -> Dict[str, Any]:
        return {
            'bilstm_dim': self.bilstm_dim,
            'embedding_dim': self.embedding_dim,
        }
