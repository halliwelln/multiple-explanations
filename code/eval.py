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
    help='paul, simpsons')
parser.add_argument('rule',type=str,
    help='spouse,uncle,...,full_data')
parser.add_argument('model',type=str)
parser.add_argument('trace_length',type=int)

args = parser.parse_args()

DATASET = args.dataset
RULE = args.rule
MODEL = args.model
TRACE_LENGTH = args.trace_length

data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,weights,entities,relations = utils.get_data(data,RULE)

UNK_ENT_ID = 'UNK_ENT'
UNK_REL_ID = 'UNK_REL'
UNK_WEIGHT_ID = 'UNK_WEIGHT'
MAX_TRACE = data['max_trace']

###################################################

if (MODEL == 'gnn_explainer') or (MODEL == 'all'):

    gnn_data = np.load(
        os.path.join('..','data','preds',DATASET,
            'gnn_explainer_'+DATASET+'_'+RULE+'_preds.npz'),allow_pickle=True)

    gnn_test_idx = gnn_data['test_idx']

    gnn_true_exps = traces[gnn_test_idx]
    gnn_true_weights = weights[gnn_test_idx]

    gnn_preds = gnn_data['preds']

    num_gnn_triples = gnn_true_exps.shape[0]

    gnn_jaccard = 0.0
    gnn_precision = 0.0
    gnn_recall = 0.0

    for i in range(num_gnn_triples):

        gnn_true_exp = gnn_true_exps[i]
        gnn_pred = gnn_preds[i]
        true_weight = gnn_true_weights[i]

        gnn_jaccard += utils.max_jaccard_np(gnn_true_exp,gnn_pred,UNK_ENT_ID,UNK_REL_ID)

        gnn_precision_i, gnn_recall_i = utils.graded_precision_recall(
            gnn_true_exp,gnn_pred,true_weight,UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

        gnn_precision += gnn_precision_i
        gnn_recall += gnn_recall_i

    gnn_jaccard /= num_gnn_triples
    gnn_precision /= num_gnn_triples
    gnn_recall /= num_gnn_triples

    gnn_f1 = utils.f1(gnn_precision,gnn_recall)

    print(f'{DATASET} {RULE} GnnExplainer')
    print(f'graded precision {round(gnn_precision,3)}')
    print(f'graded recall {round(gnn_recall,3)}')
    print(f'f1 {round(gnn_f1,3)}')
    print(f'max jaccard score: {round(gnn_jaccard,3)}')


###################################################
if (MODEL == 'explaine') or (MODEL == 'all'):

    explaine_data = np.load(
        os.path.join('..','data','preds',DATASET,
            'explaine_'+DATASET+'_'+RULE+'_preds.npz'),allow_pickle=True)

    explaine_test_idx = explaine_data['test_idx']

    explaine_true_exps = traces[explaine_test_idx]
    explaine_true_weights = weights[explaine_test_idx]

    explaine_preds = explaine_data['preds']

    num_explaine_triples = explaine_true_exps.shape[0]

    explaine_jaccard = 0.0
    explaine_precision = 0.0
    explaine_recall = 0.0

    for i in range(num_explaine_triples):

        explaine_true_exp = explaine_true_exps[i]
        explaine_pred = explaine_preds[i]
        true_weight = explaine_true_weights[i]

        explaine_jaccard += utils.max_jaccard_np(explaine_true_exp,explaine_pred,UNK_ENT_ID,UNK_REL_ID)

        explaine_precision_i, explaine_recall_i = utils.graded_precision_recall(
            explaine_true_exp,explaine_pred,true_weight,UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

        explaine_precision += explaine_precision_i
        explaine_recall += explaine_recall_i

    explaine_jaccard /= num_explaine_triples
    explaine_precision /= num_explaine_triples
    explaine_recall /= num_explaine_triples

    explaine_f1 = utils.f1(explaine_precision,explaine_recall)

    print(f'{DATASET} {RULE} ExplaiNE')
    print(f'graded precision {round(explaine_precision,3)}')
    print(f'graded recall {round(explaine_recall,3)}')
    print(f'f1 {round(explaine_f1,3)}')
    print(f'max jaccard score: {round(explaine_jaccard,3)}')


