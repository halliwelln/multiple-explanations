#!/bin/bash

#conda activate kg_env

DATASET='french_royalty'

#RULE='spouse'

#MODEL='explaine'
MODEL='gnn_explainer'

./eval.py $DATASET 'spouse' $MODEL
./eval.py $DATASET 'brother' $MODEL
./eval.py $DATASET 'sister' $MODEL
./eval.py $DATASET 'grandparent' $MODEL
./eval.py $DATASET 'child' $MODEL
./eval.py $DATASET 'parent' $MODEL
./eval.py $DATASET 'full_data' $MODEL
