#!/usr/bin/env python3

import numpy as np
import argparse
import os
import utils
import random as rn
from sklearn.model_selection import KFold
import tensorflow as tf
import RGCN

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
rn.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str,
    help='paul, french_royalty, etc')
parser.add_argument('rule',type=str,
    help='spouse,aunt,...,full_data')
parser.add_argument('embedding_dim',type=int)

args = parser.parse_args()

DATASET = args.dataset
RULE = args.rule
EMBEDDING_DIM = args.embedding_dim
OUTPUT_DIM = EMBEDDING_DIM

data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,weights,entities,relations = utils.get_data(data,RULE)

MAX_PADDING = 2
LONGEST_TRACE = utils.get_longest_trace(data, RULE)

X_train_triples, X_train_traces,_,\
        X_test_triples, X_test_traces, _ = utils.train_test_split_no_unseen(
                        X=triples,E=traces,weights=weights,
                        longest_trace=LONGEST_TRACE,max_padding=MAX_PADDING,
                        test_size=.25,seed=SEED)

NUM_ENTITIES = len(entities)
NUM_RELATIONS = len(relations)

ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

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

model.load_weights(os.path.join('..','data','weights',DATASET,DATASET + '_'+RULE+'.h5'))

train2idx = utils.array2idx(X_train_triples,ent2idx,rel2idx)
trainexp2idx = utils.array2idx(X_train_traces,ent2idx,rel2idx)

test2idx = utils.array2idx(X_test_triples,ent2idx,rel2idx)
testexp2idx = utils.array2idx(X_test_traces,ent2idx,rel2idx)

ADJACENCY_DATA = tf.concat([
    train2idx,
    trainexp2idx.reshape(-1,3),
    test2idx,
    testexp2idx.reshape(-1,3)
    ],axis=0
)

ADJ_MATS = utils.get_adj_mats(ADJACENCY_DATA,NUM_ENTITIES,NUM_RELATIONS)

X_test = np.expand_dims(test2idx,axis=0)

preds = model.predict(
    x=[
        ALL_INDICES,
        X_test[:,:,0],
        X_test[:,:,1],
        X_test[:,:,2],
        ADJ_MATS
    ]
)

acc = np.mean(preds > .5)

print(f'Embedding dim: {EMBEDDING_DIM}')
print(f'{DATASET} {RULE} accuracy {round(acc,3)}')

