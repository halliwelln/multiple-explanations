#!/bin/bash

#conda activate kg_env

DATASET='french_royalty'
RULE='full_data'
NUM_EPOCHS=2000
EMBEDDING_DIM=10
LR=.01

./RGCN.py $DATASET $RULE $NUM_EPOCHS $EMBEDDING_DIM $LR

./rgcn_eval.py $DATASET $RULE $EMBEDDING_DIM

#TOP_K=2

#./explaiNE.py $DATASET $RULE $TOP_K $EMBEDDING_DIM

#MODEL='explaine'

#./eval.py $DATASET $RULE $MODEL $TOP_K