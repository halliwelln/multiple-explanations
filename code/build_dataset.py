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
parser.add_argument("rules", type=str,
    help='<rule> separated by spaces (no quotes or commas) i.e grandparent child')
args = parser.parse_args()

rules = args.rules.split()

MAX_PADDING = 2

UNK = np.array([['UNK_ENT','UNK_REL','UNK_ENT']])
UNK_WEIGHT_STR = 'UNK_WEIGHT'
UNK_WEIGHT = np.array([[UNK_WEIGHT_STR] * MAX_PADDING])

data = dict()
all_triples = []
all_traces = []
all_weights = []

for rule in rules:

    if rule == 'uncle' or rule == 'aunt':
        rule_file = 'uncle_aunt'

    elif rule == 'brother' or rule == 'sister':
        rule_file = 'brother_sister'
        MAX_PADDING = 3
    else:
        rule_file = rule

    triples,traces, weights = utils.parse_ttl(
        file_name=os.path.join('..','data','traces',rule_file+'.ttl'),
        max_padding=MAX_PADDING
    )

    _, unique_traces_idx = np.unique(traces, axis=0,return_index=True)

    triples = triples[unique_traces_idx]
    traces = traces[unique_traces_idx]
    weights = weights[unique_traces_idx]

    # #remove gender triples from brother/sister
    if rule_file == 'brother_sister':

        gender_indices = (traces[:,:,1] == 'gender').any(axis=1)

        traces = traces[~gender_indices]

        MAX_PADDING = 2

    _, unique_triples_idx = np.unique(triples,axis=0,return_index=True)

    triple_lookup = {}
    longest_trace = -1

    for i in unique_triples_idx:
        
        triple = triples[i]
        
        indices = (triples == triple).all(axis=1)
            
        triple_lookup[str(triple)] = indices
        
        sum_indices = indices.sum()
        
        if sum_indices > longest_trace:
            
            longest_trace = sum_indices

    processed_triples = []
    processed_weights = []
    processed_traces = []

    for idx in unique_triples_idx:
        
        triple = triples[idx]
        
        trace_indices = triple_lookup[str(triple)]
        trace = traces[trace_indices]
        weight = weights[trace_indices]
        
        per_trace_weights = []

        for i in range(len(trace)):

            num_triples = trace[i].shape[0]
            current_weight = weights[trace_indices][i]

            num_unk = (trace[i] == UNK).all(axis=1).sum()

            current_weights = [current_weight] * (num_triples-num_unk)

            while len(current_weights) != num_triples:

                current_weights.append(UNK_WEIGHT_STR)
                
            per_trace_weights.append(current_weights)
              
        per_trace_weights = np.array(per_trace_weights)
        
        while per_trace_weights.shape[0] != longest_trace:
            per_trace_weights = np.concatenate([per_trace_weights,UNK_WEIGHT],axis=0)
            
        padded_trace = utils.pad_trace(trace,max_padding=MAX_PADDING,longest_trace=longest_trace,unk=UNK)
        
        processed_triples.append(triple)
        processed_traces.append(padded_trace)
        processed_weights.append(per_trace_weights)

    del triples
    del traces
    del weights

    triples = np.array(processed_triples)
    traces = np.array(processed_traces)
    weights = np.array(processed_weights)

    del processed_triples
    del processed_traces
    del processed_weights

    idx = triples[:,1] == rule

    triples = triples[idx]
    traces = traces[idx]
    weights = weights[idx]

    exp_entities = np.array([
        [traces[:,i,j,0],traces[:,i,j,2]] for i in range(longest_trace) for j in range(MAX_PADDING)]).flatten()

    exp_relations = np.array([
        [traces[:,i,j,1]] for i in range(longest_trace) for j in range(MAX_PADDING)]).flatten()
 
    all_triples.append(triples)
    all_traces.append(traces)
    all_weights.append(weights)

    data[rule + '_triples'] = triples
    data[rule + '_traces'] = traces
    data[rule + '_weights'] = weights
    data[rule + '_entities'] = np.unique(np.concatenate([triples[:,0], triples[:,2], exp_entities],axis=0))
    data[rule + '_relations'] = np.unique(np.concatenate([triples[:,1], exp_relations],axis=0))
    data[rule + '_longest_trace'] = longest_trace

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