Loading the FrenchRoyalty-200k is as simple as decompressing the $french_royalty.npz.zip$ file and using the *utils.get_data* function:

```python
DATASET = 'french_royalty'
RULE = 'spouse'

data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,weights,entities,relations = utils.get_data(data,RULE)
```
where $traces[0]$

gives an explanation for $triples[0]$

To reproduce results, first build a conda environment which uses Python 3.7 and Tensorflow-GPU 2.3:
```
conda env create -f kg_env.yml --name kg_env
```
