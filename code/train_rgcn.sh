#!/bin/bash

#conda activate kg_env_new

DATASET='paul'
RULE='spouse'

RGCN_EPOCHS=20
EMBEDDING_DIM=10
RGCN_LR=.001

./RGCN.py $DATASET $RULE $RGCN_EPOCHS $EMBEDDING_DIM $RGCN_LR