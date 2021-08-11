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
    help='paul, royalty')
parser.add_argument('rule',type=str,
    help='spouse,uncle,...,full_data')
parser.add_argument('embedding_dim',type=int)
parser.add_argument('trace_length',type=int)

args = parser.parse_args()

DATASET = args.dataset
RULE = args.rule
TRACE_LENGTH = args.trace_length

data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,weights,entities,relations = utils.get_data(data,RULE)

UNK_ENT_ID = 'UNK_ENT'
UNK_REL_ID = 'UNK_REL'
UNK_WEIGHT_ID = 'UNK_WEIGHT'
MAX_TRACE = data['max_trace']

###################################################
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
        gnn_true_exp,gnn_pred,true_weight,MAX_TRACE,UNK_ENT_ID,UNK_REL_ID,UNK_WEIGHT_ID)

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




