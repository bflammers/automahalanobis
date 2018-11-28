#!/usr/bin/env bash

source activate pytorch

# Forest cover
python3 main.py --dataset_name forest_cover
python3 main.py --mahalanobis --dataset_name forest_cover
python3 main.py --mahalanobis --distort_inputs --dataset_name forest_cover
python3 main.py --mahalanobis --distort_targets --dataset_name forest_cover

# Kdd smtp
python3 main.py --dataset_name kdd_smtp
python3 main.py --mahalanobis --dataset_name kdd_smtp
python3 main.py --mahalanobis --distort_inputs --dataset_name kdd_smtp
python3 main.py --mahalanobis --distort_targets --dataset_name kdd_smtp

# Kdd http
python3 main.py --dataset_name kdd_http
python3 main.py --mahalanobis --dataset_name kdd_http
python3 main.py --mahalanobis --distort_inputs --dataset_name kdd_http
python3 main.py --mahalanobis --distort_targets --dataset_name kdd_http

# Shuttle
python3 main.py --dataset_name shuttle
python3 main.py --mahalanobis --dataset_name shuttle
python3 main.py --mahalanobis --distort_inputs --dataset_name shuttle
python3 main.py --mahalanobis --distort_targets --dataset_name shuttle

# Exit script
exit 0