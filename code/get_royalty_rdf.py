#!/usr/bin/env python3

from SPARQLWrapper import SPARQLWrapper, XML
import os
from rdflib import Graph

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery(f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    CONSTRUCT{{
    ?x
    dbo:child ?child .
    }}
    WHERE {{
    ?x a dbo:Royalty .
    OPTIONAL {{?x dbo:child ?child }}
    }}
    """)
sparql.setReturnFormat(XML)

results = sparql.query().convert().serialize(
    destination=os.path.join('..','data','rdf','royalty_has_child'),
    format='xml'
    )

g = Graph()

g.parse(os.path.join('..','data','rdf','royalty_october'),format='xml')
g.parse(os.path.join('..','data','rdf','royalty_has_child'),format='xml')

g.serialize(destination=os.path.join('..','data','rdf','royalty_rdf'),format='xml')

print('Done.')