#!/usr/bin/env bash

source activate pytorch

# Forest cover
python3 main.py --mahalanobis False --dataset_name forest_cover
python3 main.py --mahalanobis True --dataset_name forest_cover
python3 main.py --mahalanobis True --distort_inputs True --dataset_name forest_cover
python3 main.py --mahalanobis True --distort_targets True --dataset_name forest_cover

# Kdd smtp
python3 main.py --mahalanobis False --dataset_name kdd_smtp
python3 main.py --mahalanobis True --dataset_name kdd_smtp
python3 main.py --mahalanobis True --distort_inputs True --dataset_name kdd_smtp
python3 main.py --mahalanobis True --distort_targets True --dataset_name kdd_smtp

# Kdd http
python3 main.py --mahalanobis False --dataset_name kdd_http
python3 main.py --mahalanobis True --dataset_name kdd_http
python3 main.py --mahalanobis True --distort_inputs True --dataset_name kdd_http
python3 main.py --mahalanobis True --distort_targets True --dataset_name kdd_http

# Shuttle
python3 main.py --mahalanobis False --dataset_name shuttle
python3 main.py --mahalanobis True --dataset_name shuttle
python3 main.py --mahalanobis True --distort_inputs True --dataset_name shuttle
python3 main.py --mahalanobis True --distort_targets True --dataset_name shuttle