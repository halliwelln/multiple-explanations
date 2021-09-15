#!/usr/bin/env python3

import utils

def get_counts(
    num_triples,true_triples, 
    true_exps, preds,unk_ent_id,unk_rel_id, predicate_weights):

    predicate_counts = {}
    pred_counts = {}

    for i in range(num_triples):
        
        true_triple = true_triples[i]
        true_exp = true_exps[i]
        pred = preds[i]
        
        max_jaccard,max_idx = utils.max_jaccard_np(true_exp,pred,unk_ent_id,unk_rel_id,return_idx=True)
        
        max_predicates = true_triple[1] +'_' + '_'.join([p for p in true_exp[max_idx,:,1] if p != unk_rel_id])
        
        if max_jaccard < 1:

            if max_predicates in predicate_counts:
                predicate_counts[max_predicates] += 1
            else:
                predicate_counts[max_predicates] = 1

            if max_predicates in pred_counts:
                pred_counts[max_predicates] += 1
            else:
                pred_counts[max_predicates] = 1
    
    weight_counts = {}

    for k,count in predicate_counts.items():
        
        weight = predicate_weights[k]
        
        if weight in weight_counts:
            weight_counts[weight] += count
        else:
            weight_counts[weight] = count

    return weight_counts, pred_counts

if __name__ == "__main__":

    import numpy as np
    import os
    import json
    import argparse
    import random as rn
    import collections
    import matplotlib.pyplot as plt

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str,
        help='paul, french_royalty')
    parser.add_argument('rule',type=str,
        help='spouse,uncle,...,full_data')
    parser.add_argument('model',type=str)

    args = parser.parse_args()

    DATASET = args.dataset
    RULE = args.rule
    MODEL = args. model

    data = np.load(os.path.join('..','data',DATASET+'.npz'))

    triples,traces,weights,entities,relations = utils.get_data(data,RULE)

    UNK_ENT_ID = 'UNK_ENT'
    UNK_ENT_ID = 'UNK_REL'
    UNK_WEIGHT_ID = 'UNK_WEIGHT'

    with open(os.path.join('..','data','predicate_weights.json'),'r') as f:
        predicate_weights = json.load(f)    

    if (MODEL == 'gnn_explainer') or (MODEL == 'all'):

        gnn_data = np.load(
            os.path.join('..','data','preds',DATASET,
                'gnn_explainer_'+DATASET+'_'+RULE+'_preds.npz'),allow_pickle=True)

        # gnn_data = np.load(
        #     os.path.join('..','data','preds',DATASET,
        #         'gnn_explainer_'+DATASET+'_'+RULE+'_' + str(50)+'_preds.npz'),allow_pickle=True)

        gnn_test_idx = gnn_data['test_idx']
        gnn_true_triples = triples[gnn_test_idx]
        gnn_true_exps = traces[gnn_test_idx]

        gnn_preds = gnn_data['preds']

        num_gnn_triples = gnn_true_exps.shape[0]

        gnn_counts, gnn_pred_counts = get_counts(num_gnn_triples,gnn_true_triples,
            gnn_true_exps,gnn_preds,UNK_ENT_ID,UNK_ENT_ID,predicate_weights)

        gnn_sorted_counts = collections.OrderedDict(sorted(gnn_counts.items()))

        print(f"Rule: {RULE}")
        print(f"GNNExplainer counts {gnn_sorted_counts}")
        print(f"GNNExplainer pred counts {gnn_pred_counts}")

        if RULE == 'spouse':

            keys = ['_'.join(k.split('_')[1:]) for k,_ in gnn_pred_counts.items()]
            values = list(gnn_pred_counts.values())

            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(keys,values)
            ax.set_xticklabels(labels=keys,rotation = (45), fontsize = 14)
            plt.savefig(f"../plots/{DATASET}_gnn_explainer_{RULE}_counts.pdf",bbox_inches='tight')

    ###################################################
    if (MODEL == 'explaine') or (MODEL == 'all'):

        explaine_data = np.load(
            os.path.join('..','data','preds',DATASET,
                'explaine_'+DATASET+'_'+RULE+'_preds.npz'),allow_pickle=True)

        explaine_test_idx = explaine_data['test_idx']
        explaine_true_triples = triples[explaine_test_idx]
        explaine_true_exps = traces[explaine_test_idx]
        explaine_preds = explaine_data['preds']

        num_explaine_triples = explaine_true_exps.shape[0]

        explaine_counts, explaine_pred_counts = get_counts(num_explaine_triples,
            explaine_true_triples,explaine_true_exps,explaine_preds,
            UNK_ENT_ID,UNK_ENT_ID,predicate_weights)

        explaine_sorted_counts = collections.OrderedDict(sorted(explaine_counts.items()))
        print(f"Rule: {RULE}")
        print(f"Explaine counts {explaine_sorted_counts}")
        print(f"Explaine pred counts {explaine_pred_counts}")

        if RULE == 'spouse':

            keys = ['_'.join(k.split('_')[1:]) for k,_ in explaine_pred_counts.items()]
            values = list(explaine_pred_counts.values())

            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(keys,values)
            ax.set_xticklabels(labels=keys,rotation = (45), fontsize = 14)
            plt.savefig(f"../plots/{DATASET}_explaine_{RULE}_counts.pdf",bbox_inches='tight')

