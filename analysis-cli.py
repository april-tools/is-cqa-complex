#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import pickle
import rdflib


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        res = pickle.load(f)
    return res

def load_triples(path: str) -> list:
    res = []
    with open(path, 'r') as f:
        for line in f:
            triple = line.strip().split()
            assert len(triple) == 3
            res += [tuple([int(e) for e in triple])]
    return res

def id_to_uri(symbol_id: int) -> str:
    return f"<http://example.org/{symbol_id}>"

def uri_to_id(uri: str) -> int:
    return int(uri.rsplit("/", 1)[-1])

def triples_to_graph(triples: list) -> rdflib.Graph:
    g = rdflib.Graph()
    rdf_data = ""
    for triple in triples:
        rdf_data += f'{" ".join([id_to_uri(e) for e in triple])} .\n'
    g.parse(data=rdf_data, format="nt")
    return g

def check_query(query: tuple, answers: set, graph: rdflib.Graph) -> tuple:
    nb_found_answers = 0
    nb_not_found_answers = 0

    # (0, (59,))
    source_id = query[0]
    rel_id = query[1][0]
    query_str = f"SELECT ?a WHERE {{ {id_to_uri(source_id)} {id_to_uri(rel_id)} ?a }}"
    res = graph.query(query_str)

    retrieved_answers = {uri_to_id(row.a) for row in res}

    # breakpoint()

    for ans in answers:
        if ans in retrieved_answers:
            nb_found_answers += 1
        else:
            nb_not_found_answers += 1

    return nb_found_answers, nb_not_found_answers


def main(argv):
    prefix_path = "data/FB15k-237-betae/"

    train_path = f"{prefix_path}/train.txt"
    valid_path = f"{prefix_path}/valid.txt"
    test_path = f"{prefix_path}/test.txt"

    print(f'Loading {prefix_path}/{{train, valid, test}}.txt ..')

    train_triples = load_triples(train_path)
    valid_triples = load_triples(valid_path)
    test_triples = load_triples(test_path)
    all_triples = train_triples + valid_triples + test_triples

    print(f'Loading {prefix_path}/{{id2rel, rel2id}}.pkl ..')

    id2rel = load_pickle(f"{prefix_path}/id2rel.pkl")
    rel2id = load_pickle(f"{prefix_path}/rel2id.pkl")

    print(f'Loading {prefix_path}/{{id2ent, ent2id}}.pkl ..')

    id2ent = load_pickle(f"{prefix_path}/id2ent.pkl")
    ent2id = load_pickle(f"{prefix_path}/ent2id.pkl")

    print(f'Loading {prefix_path}/{{train, valid, test}}-queries.pkl ..')

    train_queries = load_pickle(f"{prefix_path}/train-queries.pkl")
    valid_queries = load_pickle(f"{prefix_path}/valid-queries.pkl")
    test_queries = load_pickle(f"{prefix_path}/test-queries.pkl")

    print(f'Loading {prefix_path}/train-answers.pkl ..')

    train_answers = load_pickle(f"{prefix_path}/train-answers.pkl")

    print(f'Loading {prefix_path}/{{valid-easy, valid-hard, test-easy, test-hard}}-answers.pkl ..')

    valid_easy_answers = load_pickle(f"{prefix_path}/valid-easy-answers.pkl")
    valid_hard_answers = load_pickle(f"{prefix_path}/valid-hard-answers.pkl")
    test_easy_answers = load_pickle(f"{prefix_path}/test-easy-answers.pkl")
    test_hard_answers = load_pickle(f"{prefix_path}/test-hard-answers.pkl")


    train_graph = triples_to_graph(train_triples)
    complete_graph = triples_to_graph(all_triples)

    for query, answers in train_answers.items():

        print('Query', query)
        print('Asnwers', answers)

        # nb_found_answers, nb_not_found_answers = check_query(query, answers, train_graph)
        # print(nb_found_answers, nb_not_found_answers)


if __name__ == '__main__':
    main(sys.argv[1:])
