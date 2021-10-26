# User Scored Evaluation of Non-Unique Explanations forRelational Graph Convolutional Network Link Prediction on Knowledge Graphs

Loading the FrenchRoyalty-200k into Python is as simple as decompressing the `french_royalty.npz.zip` in `/data` and using the `utils.get_data` function:

```python
import os
import utils

DATASET = 'french_royalty'
RULE = 'spouse'

data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,weights,entities,relations = utils.get_data(data,RULE)
```
where `RULE` can be: `'spouse'`, `'brother'`, `'sister'`, `'grandparent'`, `'child'`, `'parent'`, or `'full_data'`. 

`traces[0]` gives an explanation for `triples[0]`, and `weights[0]` gives the weights for each explanation triple in `traces[0]`. 

`entities` and `relations` is a numpy array containing all unique entities/relations. 

To reproduce benchmark results, first build a conda environment which uses Python 3.7 and Tensorflow-GPU 2.3:
```
conda env create -f kg_env.yml --name kg_env
```

## Please use the following citation: 
```
@inproceedings{halliwell:hal-03402766,
  TITLE = {{User Scored Evaluation of Non-Unique Explanations for Relational Graph Convolutional Network Link Prediction on Knowledge Graphs}},
  AUTHOR = {Halliwell, Nicholas and Gandon, Fabien and Lecue, Freddy},
  URL = {https://hal.archives-ouvertes.fr/hal-03402766},
  BOOKTITLE = {{International Conference on Knowledge Capture}},
  ADDRESS = {Virtual Event, United States},
  YEAR = {2021},
  MONTH = Dec,
  DOI = {10.1145/3460210.3493557},
  KEYWORDS = {link prediction ; Explainable AI ; knowledge graphs ; graph neural networks ; explanation evaluation ; link prediction},
  PDF = {https://hal.archives-ouvertes.fr/hal-03402766/file/K_CAP_2021.pdf},
  HAL_ID = {hal-03402766},
  HAL_VERSION = {v1},
}
```

The latest version of the semantic reasoner can be found here: <https://project.inria.fr/corese/>
