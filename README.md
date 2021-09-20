Loading the FrenchRoyalty-200k is as simple as decompressing the `french_royalty.npz.zip` in `/data` and using the `utils.get_data` function:

```python
import numpy as np
import utils

DATASET = 'french_royalty'
RULE = 'spouse'

data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,weights,entities,relations = utils.get_data(data,RULE)
```
where `RULE` can be: `'spouse'`, `'brother'`, `'sister'`, `'grandparent'`, `'child'`, `'parent'`, or `'full_data'`. 

`traces[0]` gives an explanation for `triples[0]`, and `weights[0]` gives the weights for each explanation in `traces[0]`. 

`entities` and `relations` is a numpy array containing all unique entities/relations. 

To reproduce results, first build a conda environment which uses Python 3.7 and Tensorflow-GPU 2.3:
```
conda env create -f kg_env.yml --name kg_env
```

