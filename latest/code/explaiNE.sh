#!/bin/bash

#conda activate kg_env

DATASET='french_royalty'
EMBEDDING_DIM=10
TOP_K=2

# ./explaiNE.py $DATASET 'spouse' $TOP_K $EMBEDDING_DIM
# ./explaiNE.py $DATASET 'brother' $TOP_K $EMBEDDING_DIM
./explaiNE.py $DATASET 'sister' $TOP_K $EMBEDDING_DIM
./explaiNE.py $DATASET 'grandparent' $TOP_K $EMBEDDING_DIM
./explaiNE.py $DATASET 'child' $TOP_K $EMBEDDING_DIM
./explaiNE.py $DATASET 'parent' $TOP_K $EMBEDDING_DIM
./explaiNE.py $DATASET 'full_data' $TOP_K $EMBEDDING_DIM

