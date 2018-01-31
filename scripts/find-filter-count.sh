#!/usr/bin/env bash

python -m vico.experiment --filters 10 --output-file filters_10.csv
python -m vico.experiment --filters 20 --output-file filters_20.csv
python -m vico.experiment --filters 40 --output-file filters_40.csv
python -m vico.experiment --filters 80  --output-file filters_80.csv
