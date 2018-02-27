# Multi-Task Learning experiment for Vico Research
## Creating the necessary data input files
```
> python -m vico.scripts.long2wide -h
usage: long2wide.py [-h] path

Convert .csv file with product labels from long format (one row pr. attribute)
to wide format (one row pr. document). Adds a new file in the same directory
with the suffix _clean appended to the filename

positional arguments:
  path        Path to .csv file to convert

optional arguments:
  -h, --help  show this help message and exit
```
```
> python -m vico.scripts.download -h
usage: Download product pages [-h] [--output-folder OUTPUT_FOLDER] label_path

positional arguments:
  label_path            Path to .csv with urls

optional arguments:
  -h, --help            show this help message and exit
  --output-folder OUTPUT_FOLDER
                        output folder to download pages into
```
```
> python -m vico.scripts.read_documents -h
usage: Read HTML pages into an sqlite database [-h]
                                               [--page-folder PAGE_FOLDER]
                                               [--database-path DATABASE_PATH]
                                               label_path

positional arguments:
  label_path            path to wide format .csv with document labels

optional arguments:
  -h, --help            show this help message and exit
  --page-folder PAGE_FOLDER
  --database-path DATABASE_PATH
```
```
> python -m vico.scripts.preprocess -h
usage: Perform preprocessing of documents and create windows for structured prediction
       [-h] [--database-path DATABASE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --database-path DATABASE_PATH
                        path to sqlite database
```
```
> python -m vico.scripts.transform_embedding -h
usage: Combine and transform pretrained word vectors into a pickled dictionary
       [-h]
       database_path embedding_path pretrained_vector_folder
       transformation_matrix_folder

positional arguments:
  database_path         path to sqlite database
  embedding_path        output path to pickled embedding dictionary
  pretrained_vector_folder
                        path to folder with pretrained vectors
  transformation_matrix_folder
                        path to folder with transformation matrices

optional arguments:
  -h, --help            show this help message and exit
```
```
> ppython -m vico.scripts.html_retro_fitter -h
usage: Retrofit word vectors for HTML tags [-h]
                                           database_path embedding_path
                                           indices.path

positional arguments:
  database_path   path to sqlite database
  embedding_path  path to pickled embedding dictionary
  indices.path    output path for pickled indices dictionary

optional arguments:
  -h, --help      show this help message and exit
```
```
> python -m vico.scripts.make_windows -h
usage:
Create windows with brand and ean BIO labels
 [-h]
                                                      indices_path
                                                      database_path

positional arguments:
  indices_path   path to pickled indices dictionary
  database_path  path to sqlite database

optional arguments:
  -h, --help     show this help message and exit
```
## Training script usage
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
