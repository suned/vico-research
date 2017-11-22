#!/usr/bin/env bash

python -m vico.validate --embedding-dim 50 --output-file embedding_dim_50.csv
python -m vico.validate --embedding-dim 100 --output-file embedding_dim_100.csv
python -m vico.validate --embedding-dim 150 --output-file embedding_dim_150.csv
python -m vico.validate --embedding-dim 200 --output-file embedding_dim_200.csv
python -m vico.validate --embedding-dim 250 --output-file embedding_dim_250.csv
python -m vico.validate --embedding-dim 300 --output-file embedding_dim_300.csv
