#!/usr/bin/env python3
    
import os
import utils
import random as rn
import argparse
import numpy as np

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str,
    help='paul, french_royalty')
parser.add_argument('model',type=str)
parser.add_argument('trace_length',type=int)

args = parser.parse_args()

DATASET = args.dataset
MODEL = args.model
TRACE_LENGTH = args.trace_length

data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,weights,entities,relations = utils.get_data(data,'full_data')

UNK_ENT_ID = 'UNK_ENT'
UNK_REL_ID = 'UNK_REL'
UNK_WEIGHT_ID = 'UNK_WEIGHT'

RULES = ['spouse', 'brother', 'sister', 'grandparent', 'child', 'parent']

###################################################

if (MODEL == 'gnn_explainer') or (MODEL == 'all'):

    gnn_data = np.load(
        os.path.join('..','data','preds',DATASET,
            'gnn_explainer_'+DATASET+'_full_data_preds.npz'),allow_pickle=True)

    # gnn_data = np.load(
    #     os.path.join('..','data','preds',DATASET,
    #         'gnn_explainer_'+DATASET+'_'+RULE+'_' + str(50)+'_preds.npz'),allow_pickle=True)

    gnn_test_idx = gnn_data['test_idx']

    gnn_true_triples = triples[gnn_test_idx]
    gnn_true_exps = traces[gnn_test_idx]
    gnn_true_weights = weights[gnn_test_idx]

    gnn_preds = gnn_data['preds']

    num_gnn_triples = gnn_true_exps.shape[0]

    gnn_jaccard_list = []
    gnn_precision_list = []
    gnn_recall_list = []
    gnn_f1_list = []

    for i in range(num_gnn_triples):

        gnn_true_exp = gnn_true_exps[i]
        gnn_pred = gnn_preds[i]
        gnn_true_weight = gnn_true_weights[i]

        gnn_jaccard_i = utils.max_jaccard_np(gnn_true_exp,gnn_pred,
            gnn_true_weight,UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

        gnn_precision_i, gnn_recall_i, gnn_f1_i = utils.graded_precision_recall(
            gnn_true_exp,gnn_pred,gnn_true_weight,UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

        gnn_jaccard_list.append(gnn_jaccard_i)
        gnn_precision_list.append(gnn_precision_i)
        gnn_recall_list.append(gnn_recall_i)
        gnn_f1_list.append(gnn_f1_i)

    gnn_jaccard_list = np.array(gnn_jaccard_list)
    gnn_precision_list = np.array(gnn_precision_list)
    gnn_recall_list = np.array(gnn_recall_list)
    gnn_f1_list = np.array(gnn_f1_list)
    
    for rule in RULES:

        gnn_indices = (gnn_true_triples[:,1] == rule)

        gnn_jaccard = np.mean(gnn_jaccard_list[gnn_indices])
        gnn_precision = np.mean(gnn_precision_list[gnn_indices])
        gnn_recall = np.mean(gnn_recall_list[gnn_indices])
        gnn_f1 = np.mean(gnn_f1_list[gnn_indices])

        print(f'{DATASET} {rule} GnnExplainer')
        print(f'graded precision {round(gnn_precision,3)}')
        print(f'graded recall {round(gnn_recall,3)}')
        print(f'f1 {round(gnn_f1,3)}')
        print(f'max jaccard score: {round(gnn_jaccard,3)}')

print('###############################################')

###################################################
if (MODEL == 'explaine') or (MODEL == 'all'):

    explaine_data = np.load(
        os.path.join('..','data','preds',DATASET,
            'explaine_'+DATASET+'_full_data_preds.npz'),allow_pickle=True)

    explaine_test_idx = explaine_data['test_idx']
    explaine_true_triples = triples[explaine_test_idx]
    explaine_true_exps = traces[explaine_test_idx]
    explaine_true_weights = weights[explaine_test_idx]

    explaine_preds = explaine_data['preds']

    num_explaine_triples = explaine_true_exps.shape[0]

    explaine_jaccard_list = []
    explaine_precision_list = []
    explaine_recall_list = []
    explaine_f1_list = []

    for i in range(num_explaine_triples):

        explaine_true_exp = explaine_true_exps[i]
        explaine_pred = explaine_preds[i]
        explaine_true_weight = explaine_true_weights[i]

        explaine_jaccard_i = utils.max_jaccard_np(explaine_true_exp,explaine_pred,explaine_true_weight,
            UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

        explaine_precision_i, explaine_recall_i, explaine_f1_i = utils.graded_precision_recall(
            explaine_true_exp,explaine_pred,explaine_true_weight,UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

        explaine_jaccard_list.append(explaine_jaccard_i)
        explaine_precision_list.append(explaine_precision_i)
        explaine_recall_list.append(explaine_recall_i)
        explaine_f1_list.append(explaine_f1_i)

    explaine_jaccard_list = np.array(explaine_jaccard_list)
    explaine_precision_list = np.array(explaine_precision_list)
    explaine_recall_list = np.array(explaine_recall_list)
    explaine_f1_list = np.array(explaine_f1_list)

    for rule in RULES:

        explaine_rule_indices = (explaine_true_triples[:,1] == rule)

        explaine_jaccard =  np.mean(explaine_jaccard_list[explaine_rule_indices])
        explaine_precision = np.mean(explaine_precision_list[explaine_rule_indices])
        explaine_recall = np.mean(explaine_recall_list[explaine_rule_indices])
        explaine_f1 = np.mean(explaine_f1_list[explaine_rule_indices])

        print(f'{DATASET} {rule} ExplaiNE')
        print(f'graded precision {round(explaine_precision,3)}')
        print(f'graded recall {round(explaine_recall,3)}')
        print(f'f1 {round(explaine_f1,3)}')
        print(f'max jaccard score: {round(explaine_jaccard,3)}')
