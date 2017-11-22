# Multi-Task Learning experiment for Vico Research

usage:
```
> python -m vico.validate -h
usage: validate.py [-h] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                   [--data-dir DATA_DIR] [--folds FOLDS]
                   [--embedding-dim EMBEDDING_DIM] [--output-file OUTPUT_FILE]
                   [--output-dir OUTPUT_DIR] [--use-attributes USE_ATTRIBUTES]
                   [--epochs EPOCHS] [--patience PATIENCE]

optional arguments:
  -h, --help            show this help message and exit
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        log level
  --data-dir DATA_DIR   path to data
  --folds FOLDS         number of cross validation folds
  --embedding-dim EMBEDDING_DIM
                        word embedding dimension
  --output-file OUTPUT_FILE
                        name of .csv file in which to store validation results
  --output-dir OUTPUT_DIR
                        name of output dir
  --use-attributes USE_ATTRIBUTES
                        include html attributes
  --epochs EPOCHS       number of epochs
  --patience PATIENCE   patience parameter for early stopping
```
