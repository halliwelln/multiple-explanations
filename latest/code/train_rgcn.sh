#!/bin/bash

#conda activate kg_env

DATASET='french_royalty'

# ./RGCN.py $DATASET 'spouse' 1000 10 0.01
# ./RGCN.py $DATASET 'brother' 1000 10 0.01
# ./RGCN.py $DATASET 'sister' 1000 10 0.01
#./RGCN.py $DATASET 'grandparent' 1000 10 0.01
# ./RGCN.py $DATASET 'child' 1000 10 0.01
# ./RGCN.py $DATASET 'parent' 1000 10 0.01
./RGCN.py $DATASET 'full_data' 1000 10 0.01

# ./rgcn_eval.py $DATASET 'spouse' 10 #accuracy: 0.935
# ./rgcn_eval.py $DATASET 'brother' 10 #accuracy: 0.909
# ./rgcn_eval.py $DATASET 'sister' 10 #accuracy: 0.853
# ./rgcn_eval.py $DATASET 'grandparent' 10 #accuracy: 0.858
# ./rgcn_eval.py $DATASET 'child' 10 #accuracy: 0.792
# ./rgcn_eval.py $DATASET 'parent' 10 #accuracy: 0.838
./rgcn_eval.py $DATASET 'full_data' 10 #accuracy: 0.928

