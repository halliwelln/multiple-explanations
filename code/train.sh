#!/bin/bash

#conda activate kg_env_new

DATASET='paul_dataset'
RULE='spouse'

RGCN_EPOCHS=20
EMBEDDING_DIM=10
RGCN_LR=.001

TRACE_LENGTH=3

#./build_dataset.py $RULE

#./RGCN.py $DATASET $RULE $RGCN_EPOCHS $EMBEDDING_DIM $RGCN_LR

./explaiNE.py $DATASET $RULE $EMBEDDING_DIM $TRACE_LENGTH