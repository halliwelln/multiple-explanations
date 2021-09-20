#!/bin/bash

conda activate kg_env

#./build_dataset.py paul aunt uncle spouse brother sister grandparent child parent

./build_dataset.py french_royalty spouse brother sister grandparent child parent

