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

_, _,_,X_test_triples, X_test_traces, X_test_weights = utils.train_test_split_no_unseen(
        X=triples,E=traces,weights=weights,test_size=.3,seed=SEED)

UNK_ENT_ID = 'UNK_ENT'
UNK_REL_ID = 'UNK_REL'
UNK_WEIGHT_ID = 'UNK_WEIGHT'

NUM_TRIPLES = len(X_test_triples)

###################################################

if (MODEL == 'gnn_explainer') or (MODEL == 'all'):

    gnn_data = np.load(
        os.path.join('..','data','preds',DATASET,
            'gnn_explainer_'+DATASET+'_'+RULE+'_preds.npz'),allow_pickle=True)

    # gnn_data = np.load(
    #     os.path.join('..','data','preds',DATASET,
    #         'gnn_explainer_'+DATASET+'_'+RULE+'_' + str(50)+'_preds.npz'),allow_pickle=True)

    gnn_preds = gnn_data['preds']

    gnn_jaccard = 0.0
    gnn_precision = 0.0
    gnn_recall = 0.0
    gnn_f1 = 0.0

    for i in range(NUM_TRIPLES):

        gnn_true_exp = X_test_traces[i]
        gnn_pred = gnn_preds[i]
        gnn_true_weight = X_test_weights[i]

        gnn_jaccard_i = utils.max_jaccard_np(gnn_true_exp,gnn_pred,gnn_true_weight,
            UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

        gnn_precision_i, gnn_recall_i, gnn_f1_i = utils.graded_precision_recall(
            gnn_true_exp,gnn_pred,gnn_true_weight,UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

        gnn_precision += gnn_precision_i
        gnn_recall += gnn_recall_i
        gnn_jaccard += gnn_jaccard_i
        gnn_f1 += gnn_f1_i

    gnn_jaccard /= NUM_TRIPLES
    gnn_precision /= NUM_TRIPLES
    gnn_recall /= NUM_TRIPLES
    gnn_f1 /= NUM_TRIPLES

    #gnn_f1 = utils.f1(gnn_precision,gnn_recall)

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

    explaine_preds = explaine_data['preds']

    explaine_jaccard = 0.0
    explaine_precision = 0.0
    explaine_recall = 0.0
    explaine_f1 = 0.0

    for i in range(NUM_TRIPLES):

        explaine_true_exp = X_test_traces[i]
        explaine_pred = explaine_preds[i]
        explaine_true_weight = X_test_weights[i]

        explaine_jaccard_i = utils.max_jaccard_np(explaine_true_exp,explaine_pred,explaine_true_weight,
            UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

        explaine_precision_i, explaine_recall_i, explaine_f1_i = utils.graded_precision_recall(
            explaine_true_exp,explaine_pred,explaine_true_weight,UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

        explaine_precision += explaine_precision_i
        explaine_recall += explaine_recall_i
        explaine_jaccard += explaine_jaccard_i
        explaine_f1 += explaine_f1_i

    explaine_jaccard /= NUM_TRIPLES
    explaine_precision /= NUM_TRIPLES
    explaine_recall /= NUM_TRIPLES
    explaine_f1 /= NUM_TRIPLES

    #explaine_f1 = utils.f1(explaine_precision,explaine_recall)

    print(f'{DATASET} {RULE} ExplaiNE')
    print(f'graded precision {round(explaine_precision,3)}')
    print(f'graded recall {round(explaine_recall,3)}')
    print(f'f1 {round(explaine_f1,3)}')
    print(f'max jaccard score: {round(explaine_jaccard,3)}')


