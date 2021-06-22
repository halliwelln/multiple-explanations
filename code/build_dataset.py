#!/usr/bin/env python3

import numpy as np
import random as rn
import os
import utils
import argparse

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("rules", nargs="+",help='<rule> no quotes or commas ie grandparent child')
args = parser.parse_args()

rules = args.rules

data = dict()
all_triples = []
all_traces = []
all_weights = []

for rule in rules:

    if rule == 'uncle' or rule == 'aunt':
        rule_file = 'uncle_aunt'

    elif rule == 'brother' or rule == 'sister':
        rule_file = 'brother_sister'
    else:
        rule_file = rule

    triples,traces, weights = utils.parse_ttl(
        file_name=os.path.join('..','data','traces',rule_file+'.ttl'),
        max_padding=MAX_PADDING
    )

    _, unique_idx = np.unique(triples, axis=0,return_index=True)

    triples = triples[unique_idx]
    traces = traces[unique_idx]
    weights = weights[unique_idx]

    idx = triples[:,1] == rule

    triples = triples[idx]
    traces = traces[idx]
    weights = weights[idx]

    #####FIX THIS######
    #remove male/female triples from brother/sister
    if rule_file == 'brother_sister':
        pass

    exp_entities = np.array([[traces[:,i,:][:,0],
        traces[:,i,:][:,2]] for i in range(MAX_PADDING)]).flatten()

    exp_relations = np.array([traces[:,i,:][:,1] for i in range(MAX_PADDING)]).flatten()
 
    if rule == 'spouse':

        spouse_idx = traces[:,0,:][:,1] == 'spouse'

        symmetric_spouse_triples = triples[spouse_idx]
        symmetric_spouse_traces = traces[spouse_idx]
        symmetric_spouse_weights = weights[spouse_idx]

        non_symmetric_spouse_triples = triples[~spouse_idx]
        non_symmetric_spouse_traces = traces[~spouse_idx]
        non_symmetric_spouse_weights = weights[~spouse_idx]

        swapped_triples = symmetric_spouse_triples[:,[2,1,0]]
        swapped_traces = symmetric_spouse_traces[:,:,[2,1,0]]

        triples = np.concatenate([
            swapped_triples,
            symmetric_spouse_triples,
            non_symmetric_spouse_triples
            ], axis=0)

        traces = np.concatenate([
            swapped_traces,
            symmetric_spouse_traces,
            non_symmetric_spouse_traces
            ], axis=0)

        weights = np.concatenate([
            symmetric_spouse_weights,
            symmetric_spouse_weights,
            non_symmetric_spouse_weights
            ], axis=0)

    all_triples.append(triples)
    all_traces.append(traces)
    all_weights.append(weights)

    data[rule + '_triples'] = triples
    data[rule + '_traces'] = traces
    data[rule + '_weights'] = weights
    data[rule + '_entities'] = np.unique(np.concatenate([triples[:,0], triples[:,2], exp_entities],axis=0))
    data[rule + '_relations'] = np.unique(np.concatenate([triples[:,1], exp_relations],axis=0))

all_triples = np.concatenate(all_triples,axis=0)
print(f"all_triples shape: {all_triples.shape}")

all_traces = np.concatenate(all_traces,axis=0)
print(f"all_traces shape: {all_traces.shape}")

all_weights = np.concatenate(all_weights,axis=0)
print(f"all_weights shape: {all_weights.shape}")

all_exp_entities = np.array([[all_traces[:,i,:][:,0],
    all_traces[:,i,:][:,2]] for i in range(MAX_PADDING)]).flatten()

all_exp_relations = np.array([all_traces[:,i,:][:,1] for i in range(MAX_PADDING)]).flatten()

all_entities = np.unique(np.concatenate([all_triples[:,0], all_triples[:,2], all_exp_entities],axis=0))
all_relations = np.unique(np.concatenate([all_triples[:,1], all_exp_relations],axis=0))

data['all_entities'] = all_entities
data['all_relations'] = all_relations
data['rules'] = rules

print('Saving numpy file...')

np.savez(os.path.join('..','data',f'paul_dataset.npz'),**data)

print('Done')