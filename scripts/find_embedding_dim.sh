#!/usr/bin/env bash

python -m vico.validate --embedding-path embeddings/de-embeddings-50.pkl --output-file embedding_dim_50.csv
python -m vico.validate --embedding-path embeddings/de-embeddings-100.pkl --output-file embedding_dim_50.csv
python -m vico.validate --embedding-path embeddings/de-embeddings-150.pkl --output-file embedding_dim_50.csv
python -m vico.validate --embedding-path embeddings/de-embeddings-200.pkl --output-file embedding_dim_50.csv
python -m vico.validate --embedding-path embeddings/de-embeddings-250.pkl --output-file embedding_dim_50.csv
python -m vico.validate --embedding-path embeddings/de-embeddings-300.pkl --output-file embedding_dim_50.csv
