# Multi-Task Learning experiment for Vico Research

usage:
```
> python -m vico.experiment -h
usage: experiment.py [-h] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                     [--database-path DATABASE_PATH]
                     [--output-file OUTPUT_FILE] [--output-dir OUTPUT_DIR]
                     [--patience PATIENCE] [--embedding-path EMBEDDING_PATH]
                     [--indices-path INDICES_PATH]
                     [--targets TARGETS [TARGETS ...]] [--filters FILTERS]
                     [--n-samples N_SAMPLES] [--window-size {5,11,21}]
                     [--skip SKIP [SKIP ...]]

optional arguments:
  -h, --help            show this help message and exit
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        log level (default: INFO)
  --database-path DATABASE_PATH
                        Path to SQLite database with pre-processed data
                        (default: ../data/all_docs.sqlite)
  --output-file OUTPUT_FILE
                        name of .csv file in which to store validation results
                        (default: 2018-02-23_12:00:44.csv)
  --output-dir OUTPUT_DIR
                        name of directory in which to store validation results
                        (default: ./reports)
  --patience PATIENCE   patience parameter for early stopping. Number of
                        epochs without improvement to tolerate before
                        stopping. (default: 5)
  --embedding-path EMBEDDING_PATH
                        path to pickled pre-trained embedding dictionary to
                        load. This dictionary maps indices to word vectors
                        (default: data/transformed_embedding.pkl)
  --indices-path INDICES_PATH
                        path to pickled indices dictionary. This dictionary
                        maps words to indices. (default: data/indices.pkl)
  --targets TARGETS [TARGETS ...]
                        Name of target tasks to run in a multi-task learning
                        experiment (default: ['brand'])
  --filters FILTERS     Number of filters to include pr. filter size in the
                        convolutional layer (default: 10)
  --n-samples N_SAMPLES
                        Number of HTML pages to run the experiment on. Useful
                        for testing/debugging purposes (default: None)
  --window-size {5,11,21}
                        Size of the window to use in IO label prediction setup
                        (default: 5)
  --skip SKIP [SKIP ...]
                        Languages to skip in leave-one-language-out cross
                        validation. Useful for stopping and resuming
                        experiments (default: [])
```
