#!/usr/bin/env python3

import RGCN
import numpy as np
import argparse
import os
import utils
import random as rn
import tensorflow as tf

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
rn.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str,
    help='paul, french_royalty, etc')
parser.add_argument('embedding_dim',type=int)

args = parser.parse_args()

DATASET = args.dataset
EMBEDDING_DIM = args.embedding_dim
OUTPUT_DIM = EMBEDDING_DIM

RULE = 'full_data'
data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,_,entities,relations = utils.get_data(data,RULE)

RULES = ['spouse', 'brother', 'sister', 'grandparent', 'child', 'parent']

NUM_ENTITIES = len(entities)
NUM_RELATIONS = len(relations)

ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

triples2idx = utils.array2idx(triples,ent2idx,rel2idx)
traces2idx = utils.array2idx(traces,ent2idx,rel2idx)

full_data = np.unique(
    np.concatenate([triples2idx,traces2idx.reshape(-1,3)],axis=0),
    axis=0)

X_train,X_test = utils.train_test_split_no_unseen(
    full_data, 
    test_size=.2,
    seed=SEED, 
    allow_duplication=False, 
    filtered_test_predicates=None
)

adj_mats = utils.get_adj_mats(X_train,NUM_ENTITIES,NUM_RELATIONS)

X_train = np.expand_dims(X_train,axis=0)
X_test = np.expand_dims(X_test,axis=0)

ALL_INDICES = np.arange(NUM_ENTITIES).reshape(1,-1)

# strategy = tf.distribute.MirroredStrategy()
# print(f'Number of devices: {strategy.num_replicas_in_sync}')

# with strategy.scope():
model = RGCN.get_RGCN_Model(
    num_entities=NUM_ENTITIES,
    num_relations=NUM_RELATIONS,
    embedding_dim=EMBEDDING_DIM,
    output_dim=OUTPUT_DIM,
    seed=SEED
)

model.load_weights(os.path.join('..','data','weights',DATASET,DATASET+'_'+RULE+'.h5'))

for rule in RULES:

    rule_indices = X_test[0,:,1] == rel2idx[rule]

    X_test_rule = X_test[:,rule_indices,:]

    preds = model.predict(
        x=[
            ALL_INDICES,
            X_test_rule[:,:,0],
            X_test_rule[:,:,1],
            X_test_rule[:,:,2],
            adj_mats
        ]
    )

    acc = np.mean((preds > .5))

    print(f'{DATASET} {rule} accuracy {round(acc,3)}')
