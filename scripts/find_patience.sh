#!/usr/bin/env bash
python -m vico.validate --filters 20 --patience 5 --embedding-path embeddings/de-embeddings-250.pkl --output-file patience_5.csv
python -m vico.validate --filters 20 --patience 10 --embedding-path embeddings/de-embeddings-250.pkl --output-file patience_10.csv
python -m vico.validate --filters 20 --patience 20 --embedding-path embeddings/de-embeddings-250.pkl --output-file patience_20.csv
python -m vico.validate --filters 20 --patience 50 --embedding-path embeddings/de-embeddings-250.pkl --output-file patience_50.csv
python -m vico.validate --filters 20 --patience 100 --embedding-path embeddings/de-embeddings-250.pkl --output-file patience_100.csv
python -m vico.validate --filters 20 --patience 200 --embedding-path embeddings/de-embeddings-250.pkl --output-file patience_200.csv
