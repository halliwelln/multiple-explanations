#!/usr/bin/env python3

import numpy as np

def get_data(data,rule):

    if rule == 'full_data':

        triples = data['all_triples']
        traces = data['all_traces'] 
        weights = data['all_weights']

        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()

    else:
        triples,traces,weights = concat_triples(data, [rule])
        entities = data[rule+'_entities'].tolist()
        relations = data[rule+'_relations'].tolist()

    return triples,traces,weights,entities,relations

def concat_triples(data, rules):

    triples = []
    traces = []
    weights = []

    for rule in rules:

        triple_name = rule + '_triples'
        traces_name = rule + '_traces'
        weights_name = rule + '_weights'

        triples.append(data[triple_name])
        traces.append(data[traces_name])
        weights.append(data[weights_name])

    triples = np.concatenate(triples, axis=0)
    traces = np.concatenate(traces, axis=0)
    weights = np.concatenate(weights,axis=0)
    
    return triples, traces, weights