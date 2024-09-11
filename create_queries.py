import pickle
import os.path as osp
import numpy as np
import click
from collections import defaultdict
import random
from copy import deepcopy
import time
import pdb
import logging
import os
from datetime import datetime


def set_logger(save_path, query_name, print_on_screen=False):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(save_path, '%s.log' % (query_name))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def index_dataset(dataset_name, force=False):
    print('Indexing dataset {0}'.format(dataset_name))
    base_path = 'data/{0}/'.format(dataset_name)
    files = ['train.txt', 'valid.txt', 'test.txt']
    indexified_files = ['train_indexified.txt', 'valid_indexified.txt', 'test_indexified.txt']
    #files = ['train.txt']
    #indexified_files = ['train_indexified.txt']
    return_flag = True
    for i in range(len(indexified_files)):
        if not osp.exists(osp.join(base_path, indexified_files[i])):
            return_flag = False
            break
    if return_flag and not force:
        print("index file exists")
        return

    ent2id, rel2id, id2rel, id2ent = {}, {}, {}, {}

    entid, relid = 0, 0

    with open(osp.join(base_path, files[0])) as f:
        lines = f.readlines()
        file_len = len(lines)

    for p, indexified_p in zip(files, indexified_files):
        fw = open(osp.join(base_path, indexified_p), "w")
        with open(osp.join(base_path, p), 'r') as f:
            for i, line in enumerate(f):
                print('[%d/%d]' % (i, file_len), end='\r')
                e1, rel, e2 = line.split('\t')
                e1 = e1.strip()
                e2 = e2.strip()
                rel = rel.strip()
                rel_reverse = '-' + rel
                rel = '+' + rel
                # rel_reverse = rel+ '_reverse'

                if p == "train.txt":
                    if e1 not in ent2id.keys():
                        ent2id[e1] = entid
                        id2ent[entid] = e1
                        entid += 1

                    if e2 not in ent2id.keys():
                        ent2id[e2] = entid
                        id2ent[entid] = e2
                        entid += 1

                    if not rel in rel2id.keys():
                        rel2id[rel] = relid
                        id2rel[relid] = rel
                        assert relid % 2 == 0
                        relid += 1

                    if not rel_reverse in rel2id.keys():
                        rel2id[rel_reverse] = relid
                        id2rel[relid] = rel_reverse
                        assert relid % 2 == 1
                        relid += 1

                if e1 in ent2id.keys() and e2 in ent2id.keys() and rel in rel2id.keys():
                    fw.write("\t".join([str(ent2id[e1]), str(rel2id[rel]), str(ent2id[e2])]) + "\n")
                    fw.write("\t".join([str(ent2id[e2]), str(rel2id[rel_reverse]), str(ent2id[e1])]) + "\n")
        fw.close()

    with open(osp.join(base_path, "stats.txt"), "w") as fw:
        fw.write("numentity: " + str(len(ent2id)) + "\n")
        fw.write("numrelations: " + str(len(rel2id)))
    with open(osp.join(base_path, 'ent2id.pkl'), 'wb') as handle:
        pickle.dump(ent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'rel2id.pkl'), 'wb') as handle:
        pickle.dump(rel2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'id2ent.pkl'), 'wb') as handle:
        pickle.dump(id2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'id2rel.pkl'), 'wb') as handle:
        pickle.dump(id2rel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('num entity: %d, num relation: %d' % (len(ent2id), len(rel2id)))
    print("indexing finished!!")


def construct_graph(base_path, indexified_files):
    # knowledge graph
    # kb[e][rel] = set([e, e, e])
    ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for indexified_p in indexified_files:
        with open(osp.join(base_path, indexified_p)) as f:
            for i, line in enumerate(f):
                if len(line) == 0:
                    continue
                e1, rel, e2 = line.split('\t')
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                rel = int(rel.strip())
                ent_out[e1][rel].add(e2)
                ent_in[e2][rel].add(e1)

    return ent_in, ent_out


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


def write_links(dataset, ent_out, train_ent_out, max_ans_num, name):
    queries = defaultdict(set)
    train_answers = defaultdict(set)
    hard_answers = defaultdict(set)
    fp_answers = defaultdict(set)
    num_more_answer = 0
    for ent in ent_out:
        for rel in ent_out[ent]:
            if len(ent_out[ent][rel]) <= max_ans_num:
                queries[('e', ('r',))].add((ent, (rel,)))
                train_answers[(ent, (rel,))] = train_ent_out[ent][rel]
                hard_answers[(ent, (rel,))] = ent_out[ent][rel]
            else:
                num_more_answer += 1

    with open('./data/%s/%s-queries.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-tp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(train_answers, f)
    with open('./data/%s/%s-fn-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(hard_answers, f)
    with open('./data/%s/%s-fp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(fp_answers, f)
    print(num_more_answer)


def compute_answers_query_2p(entity, rels, ent_out1, ent_out2):
    answer_set_final = set()
    answer_set_inter = ent_out1[entity][rels[0]]
    for i, ent in enumerate(answer_set_inter):
        answer_set_final.update(ent_out2[ent][rels[1]])
    return answer_set_final


def compute_answers_query_3p(entity, rels, ent_out1, ent_out2, ent_out3):
    answer_set_final = set()
    answer_set_inter2 = set()
    answer_set_inter1 = ent_out1[entity][rels[0]]
    for i, ent in enumerate(answer_set_inter1):
        answer_set_inter2.update(ent_out2[ent][rels[1]])
    for i, ent in enumerate(answer_set_inter2):
        answer_set_final.update(ent_out3[ent][rels[2]])
    return answer_set_final


def compute_answers_query_4p(entity, rels, ent_out1, ent_out2, ent_out3, ent_out4):
    answer_set_final = set()
    answer_set_inter2 = set()
    answer_set_inter3 = set()
    answer_set_inter1 = ent_out1[entity][rels[0]]
    for i, ent in enumerate(answer_set_inter1):
        answer_set_inter2.update(ent_out2[ent][rels[1]])
    for i, ent in enumerate(answer_set_inter2):
        answer_set_inter3.update(ent_out3[ent][rels[2]])
    for i, ent in enumerate(answer_set_inter3):
        answer_set_final.update(ent_out4[ent][rels[3]])
    return answer_set_final


# rel_per_query_1p, rel_per_query_2p, rel_per_query_3p, rel_per_query_2i, rel_per_query_3i, rel_per_query_pi, rel_per_query_ip, rel_per_query_2u, rel_per_query_up= {},{},{},{},{},{},{},{},{}
# anch_per_query_1p, anch_per_query_2p, anch_per_query_3p, anch_per_query_2i, anch_per_query_3i, anch_per_query_pi, anch_per_query_ip, anch_per_query_2u, anch_per_query_up = {},{}, {}, {}, {}, {}, {}, {}, {}


def compute_nr_answers(query_structure, query, test_only_ent_out, train_ent_out, train_answer_set, test_answer_set,
                       rel_per_query, anch_per_query):
    # 2p
    if query_structure == ['e', ['r', 'r']]:
        # 0 existing 1 predicted
        # check if answer reachable with training link
        entity = query[0]
        rel1 = query[1][0]
        rel2 = query[1][1]

        reachableanswers01 = compute_answers_query_2p(entity, [rel1, rel2], train_ent_out, test_only_ent_out)
        reachableanswers10 = compute_answers_query_2p(entity, [rel1, rel2], test_only_ent_out, train_ent_out)
        reachable_answers_1p = (reachableanswers01 | reachableanswers10) - train_answer_set
        reachableanswers11 = compute_answers_query_2p(entity, [rel1, rel2], test_only_ent_out, test_only_ent_out)
        reachable_answers_2p = reachableanswers11 - reachable_answers_1p - train_answer_set
        if len(reachable_answers_2p) > 0:
            if rel1 in rel_per_query:
                rel_per_query[rel1] += len(reachable_answers_2p)
            else:
                rel_per_query[rel1] = len(reachable_answers_2p)
            if rel2 != rel1:
                if rel2 in rel_per_query:
                    rel_per_query[rel2] += len(reachable_answers_2p)
                else:
                    rel_per_query[rel2] = len(reachable_answers_2p)

            if entity in anch_per_query:
                anch_per_query[entity] += len(reachable_answers_2p)
            else:
                anch_per_query[entity] = len(reachable_answers_2p)

        return reachable_answers_2p, rel_per_query, anch_per_query, [rel1, rel2], [entity]
    # 3p
    if query_structure == ['e', ['r', 'r', 'r']]:
        reachable_answers_3p = set()

        entity = query[0]
        rel1 = query[1][0]
        rel2 = query[1][1]
        rel3 = query[1][2]

        # existing + existing + predicted --> 1p 001
        reachable_answers_001 = compute_answers_query_3p(entity, [rel1, rel2, rel3], train_ent_out, train_ent_out,
                                                         test_only_ent_out)
        # existing + predicted + existing --> 1p 010
        reachable_answers_010 = compute_answers_query_3p(entity, [rel1, rel2, rel3], train_ent_out, test_only_ent_out,
                                                         train_ent_out)
        # predicted + existing + existing --> 1p 100
        reachable_answers_100 = compute_answers_query_3p(entity, [rel1, rel2, rel3], test_only_ent_out, train_ent_out,
                                                         train_ent_out)
        reachable_answers_1p = (
                                           reachable_answers_001 | reachable_answers_010 | reachable_answers_100) - train_answer_set  # subtract the train answers

        if len(reachable_answers_1p) < len(test_answer_set):
            # continue the computation for 2p/3p
            # existing + predicted + predicted --> 2p 011
            reachable_answers_011 = compute_answers_query_3p(entity, [rel1, rel2, rel3], train_ent_out,
                                                             test_only_ent_out, test_only_ent_out)
            # predicted + predicted + existing --> 2p 110
            reachable_answers_110 = compute_answers_query_3p(entity, [rel1, rel2, rel3], test_only_ent_out,
                                                             test_only_ent_out, train_ent_out)
            # predicted + existing + predicted --> 2p 101
            reachable_answers_101 = compute_answers_query_3p(entity, [rel1, rel2, rel3], test_only_ent_out,
                                                             train_ent_out, test_only_ent_out)
            reachable_answers_2p = (
                                               reachable_answers_011 | reachable_answers_110 | reachable_answers_101) - reachable_answers_1p - train_answer_set  # subtract the train answers and the 1p answers
            if len(reachable_answers_1p | reachable_answers_2p) < len(test_answer_set):
                # predicted + predicted + existing --> 3p 111
                reachable_answers_111 = compute_answers_query_3p(entity, [rel1, rel2, rel3], test_only_ent_out,
                                                                 test_only_ent_out, test_only_ent_out)
                reachable_answers_3p = reachable_answers_111 - reachable_answers_2p - reachable_answers_1p - train_answer_set  # subtract the train answers and the 1p/2p answers
        if len(reachable_answers_3p) > 0:
            if rel1 in rel_per_query:
                rel_per_query[rel1] += len(reachable_answers_3p)
            else:
                rel_per_query[rel1] = len(reachable_answers_3p)
            if rel1 != rel2:
                if rel2 in rel_per_query:
                    rel_per_query[rel2] += len(reachable_answers_3p)
                else:
                    rel_per_query[rel2] = len(reachable_answers_3p)
            if rel3 != rel2 and rel3 != rel1:
                if rel3 in rel_per_query:
                    rel_per_query[rel3] += len(reachable_answers_3p)
                else:
                    rel_per_query[rel3] = len(reachable_answers_3p)

            if entity in anch_per_query:
                anch_per_query[entity] += len(reachable_answers_3p)
            else:
                anch_per_query[entity] = len(reachable_answers_3p)

        return reachable_answers_3p, rel_per_query, anch_per_query, [rel1, rel2, rel3], [entity]

        # 4p

    # 4p
    if query_structure == ['e', ['r', 'r', 'r', 'r']]:
        reachable_answers_3p = set()
        reachable_answers_4p = set()
        entity = query[0]
        rel1 = query[1][0]
        rel2 = query[1][1]
        rel3 = query[1][2]
        rel4 = query[1][3]

        # 1p 0001
        reachable_answers_0001 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], train_ent_out,
                                                          train_ent_out, train_ent_out, test_only_ent_out)
        # 0010
        reachable_answers_0010 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], train_ent_out,
                                                          train_ent_out,
                                                          test_only_ent_out, train_ent_out)
        # 0100
        reachable_answers_0100 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], train_ent_out,
                                                          test_only_ent_out,
                                                          train_ent_out, train_ent_out)
        # 1000
        reachable_answers_1000 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], test_only_ent_out,
                                                          train_ent_out,
                                                          train_ent_out, train_ent_out)
        reachable_answers_1p = (reachable_answers_0001 | reachable_answers_0010 | reachable_answers_0100 | reachable_answers_1000) - train_answer_set  # subtract the train answers

        if len(reachable_answers_1p) < len(test_answer_set):
            # 2p
            # 0011
            reachable_answers_0011 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], train_ent_out,
                                                              train_ent_out, test_only_ent_out, test_only_ent_out)
            # 0101
            reachable_answers_0101 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], train_ent_out,
                                                              test_only_ent_out, train_ent_out, test_only_ent_out)
            # 0110
            reachable_answers_0110 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], train_ent_out,
                                                              test_only_ent_out, test_only_ent_out, train_ent_out)
            # 1001
            reachable_answers_1001 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], test_only_ent_out,
                                                              train_ent_out, train_ent_out, test_only_ent_out)
            # 1010
            reachable_answers_1010 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], test_only_ent_out,
                                                              train_ent_out, test_only_ent_out, train_ent_out)
            # 1100
            reachable_answers_1100 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], test_only_ent_out,
                                                              test_only_ent_out, train_ent_out, train_ent_out)

            reachable_answers_2p = (reachable_answers_0011 | reachable_answers_0101 | reachable_answers_0110 | reachable_answers_1001 | reachable_answers_1010 | reachable_answers_1100) - reachable_answers_1p - train_answer_set  # subtract the train answers and the 1p answers
            if len(reachable_answers_1p | reachable_answers_2p) < len(test_answer_set):
                # 3p
                # 0111
                reachable_answers_0111 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], train_ent_out,
                                                                  test_only_ent_out, test_only_ent_out,
                                                                  test_only_ent_out)
                # 1011
                reachable_answers_1011 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], test_only_ent_out,
                                                                  train_ent_out, test_only_ent_out,
                                                                  test_only_ent_out)
                # 1101
                reachable_answers_1101 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], test_only_ent_out,
                                                                  test_only_ent_out, train_ent_out,
                                                                  test_only_ent_out)
                # 1110
                reachable_answers_1110 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], test_only_ent_out,
                                                                  test_only_ent_out, test_only_ent_out,
                                                                  train_ent_out)
                reachable_answers_3p = (reachable_answers_0111 | reachable_answers_1011 | reachable_answers_1101 | reachable_answers_1110) - reachable_answers_2p - reachable_answers_1p - train_answer_set  # subtract the train answers and the 1p/2p answers
                if len(reachable_answers_1p | reachable_answers_2p | reachable_answers_3p) < len(test_answer_set):
                    reachable_answers_1111 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4],
                                                                      test_only_ent_out,
                                                                      test_only_ent_out, test_only_ent_out,
                                                                      test_only_ent_out)

                    reachable_answers_4p = reachable_answers_1111 - reachable_answers_3p - reachable_answers_2p - reachable_answers_1p - train_answer_set  # subtract the train answers and the 1p/2p answers
        if len(reachable_answers_4p) > 0:
            if rel1 in rel_per_query:
                rel_per_query[rel1] += len(reachable_answers_4p)
            else:
                rel_per_query[rel1] = len(reachable_answers_4p)
            if rel1 != rel2:
                if rel2 in rel_per_query:
                    rel_per_query[rel2] += len(reachable_answers_4p)
                else:
                    rel_per_query[rel2] = len(reachable_answers_4p)
            if rel3 != rel2 and rel3 != rel1:
                if rel3 in rel_per_query:
                    rel_per_query[rel3] += len(reachable_answers_4p)
                else:
                    rel_per_query[rel3] = len(reachable_answers_4p)

            if entity in anch_per_query:
                anch_per_query[entity] += len(reachable_answers_4p)
            else:
                anch_per_query[entity] = len(reachable_answers_4p)

        return reachable_answers_4p, rel_per_query, anch_per_query, [rel1, rel2, rel3, rel4], [entity]

    # 2i
    if query_structure == [['e', ['r']], ['e', ['r']]]:
        reachable_answers_2i = set()
        # 2i
        entity1 = query[0][0]
        rel1 = query[0][1][0]
        entity2 = query[1][0]
        rel2 = query[1][1][0]

        # 01
        # compute the answers of the query (entity1,rel1,?y) on the training graph
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        # compute the answers of the query (entity2,rel2,?y) on the test_only graph
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        answers_01 = answer_set_q1_1 & answer_set_q1_2
        # 10
        # compute the answers of the query (entity1,rel1,?y) on the test_only graph
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        # compute the answers of the query (entity2,rel2,?y) on the training graph
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        answers_10 = answer_set_q1_1 & answer_set_q1_2

        reachable_answers_1p = (answers_01 | answers_10) - train_answer_set

        if len(reachable_answers_1p) < len(test_answer_set):
            # 11
            # compute the answers of the query (entity1,rel1,?y) on the test_only graph
            answer_set_q1_1 = test_only_ent_out[entity1][rel1]
            # compute the answers of the query (entity2,rel2,?y) on the test_only graph
            answer_set_q1_2 = test_only_ent_out[entity2][rel2]
            answers_11 = answer_set_q1_1 & answer_set_q1_2
            reachable_answers_2i = answers_11 - reachable_answers_1p - train_answer_set

        if len(reachable_answers_2i) > 0:
            if rel1 in rel_per_query:
                rel_per_query[rel1] += len(reachable_answers_2i)
            else:
                rel_per_query[rel1] = len(reachable_answers_2i)
            if rel1 != rel2:
                if rel2 in rel_per_query:
                    rel_per_query[rel2] += len(reachable_answers_2i)
                else:
                    rel_per_query[rel2] = len(reachable_answers_2i)

            if entity1 in anch_per_query:
                anch_per_query[entity1] += len(reachable_answers_2i)
            else:
                anch_per_query[entity1] = len(reachable_answers_2i)
            if entity1 != entity2:
                if entity2 in anch_per_query:
                    anch_per_query[entity2] += len(reachable_answers_2i)
                else:
                    anch_per_query[entity2] = len(reachable_answers_2i)
        return reachable_answers_2i, rel_per_query, anch_per_query, [rel1, rel2], [entity1, entity2]
    # 3i
    if query_structure == [['e', ['r']], ['e', ['r']], ['e', ['r']]]:
        reachable_answers_3i = set()
        # 3i
        entity1 = query[0][0]
        rel1 = query[0][1][0]
        entity2 = query[1][0]
        rel2 = query[1][1][0]
        entity3 = query[2][0]
        rel3 = query[2][1][0]

        # 001
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        answer_set_q1_3 = test_only_ent_out[entity3][rel3]
        answers_001 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3
        # 010
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        answer_set_q1_3 = train_ent_out[entity3][rel3]
        answers_010 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3
        # 100
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        answer_set_q1_3 = train_ent_out[entity3][rel3]
        answers_100 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3
        reachable_answers_1p = (answers_001 | answers_010 | answers_100) - train_answer_set

        if len(reachable_answers_1p) < len(test_answer_set):
            # 011
            answer_set_q1_1 = train_ent_out[entity1][rel1]
            answer_set_q1_2 = test_only_ent_out[entity2][rel2]
            answer_set_q1_3 = test_only_ent_out[entity3][rel3]
            answers_011 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3

            # 101
            answer_set_q1_1 = test_only_ent_out[entity1][rel1]
            answer_set_q1_2 = train_ent_out[entity2][rel2]
            answer_set_q1_3 = test_only_ent_out[entity3][rel3]
            answers_101 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3

            # 110
            answer_set_q1_1 = test_only_ent_out[entity1][rel1]
            answer_set_q1_2 = test_only_ent_out[entity2][rel2]
            answer_set_q1_3 = train_ent_out[entity3][rel3]
            answers_110 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3

            reachable_answers_2i = (answers_011 | answers_101 | answers_110) - reachable_answers_1p - train_answer_set

            if len(reachable_answers_2i | reachable_answers_1p) < len(test_answer_set):
                # 111
                answer_set_q1_1 = test_only_ent_out[entity1][rel1]
                answer_set_q1_2 = test_only_ent_out[entity2][rel2]
                answer_set_q1_3 = test_only_ent_out[entity3][rel3]
                answers_111 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3
                reachable_answers_3i = (answers_111 - reachable_answers_1p - reachable_answers_2i) - train_answer_set

        if len(reachable_answers_3i) > 0:
            rel_per_query[rel1] = len(reachable_answers_3i)
            if rel1 != rel2:
                rel_per_query[rel2] = len(reachable_answers_3i)
            if rel1 != rel3 and rel2 != rel3:
                rel_per_query[rel3] = len(reachable_answers_3i)

                anch_per_query[entity1] = len(reachable_answers_3i)
            if entity1 != entity2:
                if entity2 in anch_per_query:
                    anch_per_query[entity2] += len(reachable_answers_3i)
                else:
                    anch_per_query[entity2] = len(reachable_answers_3i)
            if entity1 != entity3 and entity2 != entity3:
                if entity3 in anch_per_query:
                    anch_per_query[entity3] += len(reachable_answers_3i)
                else:
                    anch_per_query[entity3] = len(reachable_answers_3i)

        return reachable_answers_3i, rel_per_query, anch_per_query, [rel1, rel2, rel3], [entity1, entity2, entity3]

    # 4i
    if query_structure == [['e', ['r']], ['e', ['r']], ['e', ['r']], ['e', ['r']]]:
        reachable_answers_4i = set()
        # 4i
        entity1 = query[0][0]
        rel1 = query[0][1][0]
        entity2 = query[1][0]
        rel2 = query[1][1][0]
        entity3 = query[2][0]
        rel3 = query[2][1][0]
        entity4 = query[3][0]
        rel4 = query[3][1][0]

        # 0001
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        answer_set_q1_3 = train_ent_out[entity3][rel3]
        answer_set_q1_4 = test_only_ent_out[entity4][rel4]
        answers_0001 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4
        # 0010
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        answer_set_q1_3 = test_only_ent_out[entity3][rel3]
        answer_set_q1_4 = train_ent_out[entity4][rel4]
        answers_0010 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4
        # 0100
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        answer_set_q1_3 = train_ent_out[entity3][rel3]
        answer_set_q1_4 = train_ent_out[entity4][rel4]
        answers_0100 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

        # 1000
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        answer_set_q1_3 = train_ent_out[entity3][rel3]
        answer_set_q1_4 = train_ent_out[entity4][rel4]
        answers_1000 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4




        reachable_answers_1p = (answers_0001 | answers_0010 | answers_0100 | answers_1000) - train_answer_set

        if len(reachable_answers_1p) < len(test_answer_set):

            # 0011
            answer_set_q1_1 = train_ent_out[entity1][rel1]
            answer_set_q1_2 = train_ent_out[entity2][rel2]
            answer_set_q1_3 = test_only_ent_out[entity3][rel3]
            answer_set_q1_4 = test_only_ent_out[entity4][rel4]
            answers_0011 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

            # 0101
            answer_set_q1_1 = train_ent_out[entity1][rel1]
            answer_set_q1_2 = test_only_ent_out[entity2][rel2]
            answer_set_q1_3 = train_ent_out[entity3][rel3]
            answer_set_q1_4 = test_only_ent_out[entity4][rel4]
            answers_0101 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

            # 0110
            answer_set_q1_1 = train_ent_out[entity1][rel1]
            answer_set_q1_2 = test_only_ent_out[entity2][rel2]
            answer_set_q1_3 = test_only_ent_out[entity3][rel3]
            answer_set_q1_4 = train_ent_out[entity4][rel4]
            answers_0110 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

            # 1001
            answer_set_q1_1 = test_only_ent_out[entity1][rel1]
            answer_set_q1_2 = train_ent_out[entity2][rel2]
            answer_set_q1_3 = train_ent_out[entity3][rel3]
            answer_set_q1_4 = test_only_ent_out[entity4][rel4]
            answers_1001 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

            # 1010
            answer_set_q1_1 = test_only_ent_out[entity1][rel1]
            answer_set_q1_2 = train_ent_out[entity2][rel2]
            answer_set_q1_3 = test_only_ent_out[entity3][rel3]
            answer_set_q1_4 = train_ent_out[entity4][rel4]
            answers_1010 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

            # 1100
            answer_set_q1_1 = test_only_ent_out[entity1][rel1]
            answer_set_q1_2 = test_only_ent_out[entity2][rel2]
            answer_set_q1_3 = train_ent_out[entity3][rel3]
            answer_set_q1_4 = train_ent_out[entity4][rel4]
            answers_1100 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

            reachable_answers_2i = (answers_0011 | answers_0101 | answers_0110 | answers_1001 | answers_1010 | answers_1100) - reachable_answers_1p - train_answer_set

            if len(reachable_answers_2i | reachable_answers_1p) < len(test_answer_set):

                # 0111
                answer_set_q1_1 = train_ent_out[entity1][rel1]
                answer_set_q1_2 = test_only_ent_out[entity2][rel2]
                answer_set_q1_3 = test_only_ent_out[entity3][rel3]
                answer_set_q1_4 = test_only_ent_out[entity4][rel4]
                answers_0111 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                # 1011
                answer_set_q1_1 = test_only_ent_out[entity1][rel1]
                answer_set_q1_2 = train_ent_out[entity2][rel2]
                answer_set_q1_3 = test_only_ent_out[entity3][rel3]
                answer_set_q1_4 = test_only_ent_out[entity4][rel4]
                answers_1011 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4
                # 1101
                answer_set_q1_1 = test_only_ent_out[entity1][rel1]
                answer_set_q1_2 = test_only_ent_out[entity2][rel2]
                answer_set_q1_3 = train_ent_out[entity3][rel3]
                answer_set_q1_4 = test_only_ent_out[entity4][rel4]
                answers_1101 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4
                # 1110
                answer_set_q1_1 = test_only_ent_out[entity1][rel1]
                answer_set_q1_2 = test_only_ent_out[entity2][rel2]
                answer_set_q1_3 = test_only_ent_out[entity3][rel3]
                answer_set_q1_4 = train_ent_out[entity4][rel4]
                answers_1110 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                reachable_answers_3i = (answers_0111| answers_1011|answers_1101 |answers_1110) - reachable_answers_1p - reachable_answers_2i - train_answer_set


                if len(reachable_answers_3i |reachable_answers_2i | reachable_answers_1p) < len(test_answer_set):
                    # 1111
                    answer_set_q1_1 = test_only_ent_out[entity1][rel1]
                    answer_set_q1_2 = test_only_ent_out[entity2][rel2]
                    answer_set_q1_3 = test_only_ent_out[entity3][rel3]
                    answer_set_q1_4 = test_only_ent_out[entity4][rel4]
                    answers_1111 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                    reachable_answers_4i = answers_1111 - reachable_answers_1p - reachable_answers_2i- reachable_answers_3i - train_answer_set


        if len(reachable_answers_4i) > 0:
            rel_per_query[rel1] = len(reachable_answers_4i)
            if rel1 != rel2:
                rel_per_query[rel2] = len(reachable_answers_4i)
            if rel1 != rel3 and rel2 != rel3:
                rel_per_query[rel3] = len(reachable_answers_4i)

                anch_per_query[entity1] = len(reachable_answers_4i)
            if entity1 != entity2:
                if entity2 in anch_per_query:
                    anch_per_query[entity2] += len(reachable_answers_4i)
                else:
                    anch_per_query[entity2] = len(reachable_answers_4i)
            if entity1 != entity3 and entity2 != entity3:
                if entity3 in anch_per_query:
                    anch_per_query[entity3] += len(reachable_answers_4i)
                else:
                    anch_per_query[entity3] = len(reachable_answers_4i)

        return reachable_answers_4i, rel_per_query, anch_per_query, [rel1, rel2, rel3, rel4], [entity1, entity2, entity3, entity4]

    # pi
    if query_structure == [['e', ['r', 'r']], ['e', ['r']]]:
        reachable_answers_2i = set()
        reachable_answers_2p = set()
        reachable_answers_pi = set()
        entity1 = query[0][0]
        rel1 = query[0][1][0]
        rel2 = query[0][1][1]
        entity2 = query[1][0]
        rel3 = query[1][1][0]

        # 001
        subquery_2p_answers_00 = compute_answers_query_2p(entity1, [rel1, rel2], train_ent_out, train_ent_out)

        subquery_1p_answers_1 = test_only_ent_out[entity2][rel3]
        answers_001 = subquery_2p_answers_00 & subquery_1p_answers_1
        # 010
        subquery_2p_answers_01 = compute_answers_query_2p(entity1, [rel1, rel2], train_ent_out, test_only_ent_out)
        subquery_1p_answers_0 = train_ent_out[entity2][rel3]
        answers_010 = subquery_2p_answers_01 & subquery_1p_answers_0
        # 100
        subquery_2p_answers_10 = compute_answers_query_2p(entity1, [rel1, rel2], test_only_ent_out, train_ent_out)
        answers_100 = subquery_2p_answers_10 & subquery_1p_answers_0
        reachable_answers_1p = (answers_001 | answers_010 | answers_100) - train_answer_set

        if len(reachable_answers_1p) < len(test_answer_set):
            # 2i
            # 011
            answers_011 = subquery_2p_answers_01 & subquery_1p_answers_1
            # 101
            answers_101 = subquery_2p_answers_10 & subquery_1p_answers_1
            reachable_answers_2i = (answers_011 | answers_101) - reachable_answers_1p - train_answer_set

            # 2p
            # 110
            subquery_2p_answers_11 = compute_answers_query_2p(entity1, [rel1, rel2], test_only_ent_out,
                                                              test_only_ent_out)
            answers_110 = subquery_2p_answers_11 & subquery_1p_answers_0
            reachable_answers_2p = answers_110 - reachable_answers_1p - reachable_answers_2i - train_answer_set

            if len(reachable_answers_2p | reachable_answers_2i | reachable_answers_1p) < len(test_answer_set):
                # 111
                # pi
                answers_111 = subquery_2p_answers_11 & subquery_1p_answers_1
                reachable_answers_pi = answers_111 - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i - train_answer_set

        if len(reachable_answers_pi) > 0:
            if rel1 in rel_per_query:
                rel_per_query[rel1] += len(reachable_answers_pi)
            else:
                rel_per_query[rel1] = len(reachable_answers_pi)
            if rel1 != rel2:
                if rel2 in rel_per_query:
                    rel_per_query[rel2] += len(reachable_answers_pi)
                else:
                    rel_per_query[rel2] = len(reachable_answers_pi)
            if rel1 != rel3 and rel2 != rel3:
                if rel3 in rel_per_query:
                    rel_per_query[rel3] += len(reachable_answers_pi)
                else:
                    rel_per_query[rel3] = len(reachable_answers_pi)

            if entity1 in anch_per_query:
                anch_per_query[entity1] += len(reachable_answers_pi)
            else:
                anch_per_query[entity1] = len(reachable_answers_pi)
            if entity1 != entity2:
                if entity2 in anch_per_query:
                    anch_per_query[entity2] += len(reachable_answers_pi)
                else:
                    anch_per_query[entity2] = len(reachable_answers_pi)
        return reachable_answers_pi, rel_per_query, anch_per_query, [rel1, rel2, rel3], [entity1, entity2]
    # ip
    if query_structure == [[['e', ['r']], ['e', ['r']]], ['r']]:
        reachable_answers_2i = set()
        reachable_answers_2p = set()
        reachable_answers_ip = set()
        entity1 = query[0][0][0]
        rel1 = query[0][0][1][0]
        entity2 = query[0][1][0]
        rel2 = query[0][1][1][0]
        rel3 = query[1][0]

        # 001
        answers_001 = set()
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]

        answers_00 = answer_set_q1_1 & answer_set_q1_2
        n_intermediate_ext_answers = len(answers_00)
        for ele in answers_00:
            answers_001.update(test_only_ent_out[ele][rel3])
        # 010
        answers_010 = set()
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        answers_01 = answer_set_q1_1 & answer_set_q1_2
        for ele in answers_01:
            answers_010.update(train_ent_out[ele][rel3])
        # 100
        answers_100 = set()
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        answers_10 = answer_set_q1_1 & answer_set_q1_2
        for ele in answers_10:
            answers_100.update(train_ent_out[ele][rel3])
        reachable_answers_1p = (answers_001 | answers_010 | answers_100) - train_answer_set

        if len(reachable_answers_1p) < len(test_answer_set):
            # compute 2p and 2i
            # 2i
            # 110
            answers_110 = set()
            answer_set_q1_1 = test_only_ent_out[entity1][rel1]
            answer_set_q1_2 = test_only_ent_out[entity2][rel2]
            answers_11 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_11:
                answers_110.update(train_ent_out[ele][rel3])
            reachable_answers_2i = answers_110 - reachable_answers_1p - train_answer_set
            # 2p
            # 101
            answers_101 = set()
            answer_set_q1_1 = test_only_ent_out[entity1][rel1]
            answer_set_q1_2 = train_ent_out[entity2][rel2]
            answers_10 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_10:
                answers_101.update(test_only_ent_out[ele][rel3])

            # 011
            answers_011 = set()
            answer_set_q1_1 = train_ent_out[entity1][rel1]
            answer_set_q1_2 = test_only_ent_out[entity2][rel2]
            answers_01 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_01:
                answers_011.update(test_only_ent_out[ele][rel3])
            reachable_answers_2p = (
                                               answers_101 | answers_011) - reachable_answers_1p - reachable_answers_2i - train_answer_set

            if len(reachable_answers_2p | reachable_answers_2i | reachable_answers_1p) < len(test_answer_set):
                # 111
                answers_111 = set()
                answer_set_q1_1 = test_only_ent_out[entity1][rel1]
                answer_set_q1_2 = test_only_ent_out[entity2][rel2]
                answers_11 = answer_set_q1_1 & answer_set_q1_2
                for ele in answers_11:
                    answers_111.update(test_only_ent_out[ele][rel3])
                reachable_answers_ip = answers_111 - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i - train_answer_set

        if len(reachable_answers_ip) > 0:
            if rel1 in rel_per_query:
                rel_per_query[rel1] += len(reachable_answers_ip)
            else:
                rel_per_query[rel1] = len(reachable_answers_ip)
            if rel1 != rel2:
                if rel2 in rel_per_query:
                    rel_per_query[rel2] += len(reachable_answers_ip)
                else:
                    rel_per_query[rel2] = len(reachable_answers_ip)
            if rel1 != rel3 and rel2 != rel3:
                if rel3 in rel_per_query:
                    rel_per_query[rel3] += len(reachable_answers_ip)
                else:
                    rel_per_query[rel3] = len(reachable_answers_ip)

            if entity1 in anch_per_query:
                anch_per_query[entity1] += len(reachable_answers_ip)
            else:
                anch_per_query[entity1] = len(reachable_answers_ip)
            if entity1 != entity2:
                if entity2 in anch_per_query:
                    anch_per_query[entity2] += len(reachable_answers_ip)
                else:
                    anch_per_query[entity2] = len(reachable_answers_ip)

        return reachable_answers_ip, rel_per_query, anch_per_query, [rel1, rel2, rel3], [entity1, entity2]
    # up
    if query_structure == [[['e', ['r']], ['e', ['r']], ['u']], ['r']]:
        reachable_answers_2p = set()
        reachable_answers_2u = set()
        reachable_answers_up = set()
        entity1 = query[0][0][0]
        rel1 = query[0][0][1][0]
        entity2 = query[0][1][0]
        rel2 = query[0][1][1][0]
        rel3 = query[1][0]
        '''
        answers_001 = set()
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        n_intermediate_ext_answers = max(len(answer_set_q1_1), len(answer_set_q1_2))
        answers_00 = answer_set_q1_1 | answer_set_q1_2
        for ele in answers_00:
            answers_001.update(test_only_ent_out[ele][rel3])
        # 010
        answers_010 = set()
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        answers_01 = answer_set_q1_1 | answer_set_q1_2
        for ele in answers_01:
            answers_010.update(train_ent_out[ele][rel3])
        # 100
        answers_100 = set()
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        answers_10 = answer_set_q1_1 | answer_set_q1_2
        for ele in answers_10:
            answers_100.update(train_ent_out[ele][rel3])
        # 2u is already included in 1p
        answers_110 = set()
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        answers_11 = answer_set_q1_1 | answer_set_q1_2
        for ele in answers_11:
            answers_110.update(train_ent_out[ele][rel3])

        reachable_answers_1p = (answers_001 | answers_010 | answers_100 | answers_110) - train_answer_set
        if len(reachable_answers_1p) < len(test_answer_set):
            # 2p
            # 101
            answers_101 = set()
            answer_set_q1_1 = test_only_ent_out[entity1][rel1]
            answer_set_q1_2 = train_ent_out[entity2][rel2]
            answers_10 = answer_set_q1_1 | answer_set_q1_2
            for ele in answers_10:
                answers_101.update(test_only_ent_out[ele][rel3])
            # 011
            answers_011 = set()
            answer_set_q1_1 = train_ent_out[entity1][rel1]
            answer_set_q1_2 = test_only_ent_out[entity2][rel2]
            answers_01 = answer_set_q1_1 | answer_set_q1_2
            for ele in answers_01:
                answers_011.update(test_only_ent_out[ele][rel3])
            #reachable_answers_2p = (answers_101 | answers_011) - reachable_answers_1p - reachable_answers_2u - train_answer_set
             # 111
            answers_111 = set()
            answer_set_q1_1 = test_only_ent_out[entity1][rel1]
            answer_set_q1_2 = test_only_ent_out[entity2][rel2]
            answers_11 = answer_set_q1_1 | answer_set_q1_2
            for ele in answers_11:
                answers_111.update(test_only_ent_out[ele][rel3])
            reachable_answers_2p = (answers_111 | answers_101 | answers_011) - reachable_answers_1p - train_answer_set
        '''

        # 010
        # If I can reach the same entity  of the '2u' subgraph with both a predicted link and an existing link than
        # this query,a is a 0p since there is a union. To do that I check the entities that are reachable by both link
        # and that lead to an answer (those should already be contained in the easy answers)
        answers_010 = set()
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        n_intermediate_ext_answers = max(len(answer_set_q1_1), len(answer_set_q1_2))
        answers_01 = answer_set_q1_1 & answer_set_q1_2
        for ele in answers_01:
            answers_010.update(train_ent_out[ele][rel3])
        # 100
        answers_100 = set()
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        n_intermediate_ext_answers = max(len(answer_set_q1_1), len(answer_set_q1_2))
        answers_10 = answer_set_q1_1 & answer_set_q1_2
        for ele in answers_10:
            answers_100.update(train_ent_out[ele][rel3])
        reachable_answers_0p = (answers_010 | answers_100) - train_answer_set  # should be zero
        # n_up_0p += len(reachable_answers_0p)

        # 001 #1p reduced
        answers_001 = set()
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        n_intermediate_ext_answers = max(len(answer_set_q1_1), len(answer_set_q1_2))
        answers_00 = answer_set_q1_1 & answer_set_q1_2
        for ele in answers_00:
            answers_001.update(test_only_ent_out[ele][rel3])

        answers_101 = set()
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        n_intermediate_ext_answers = max(len(answer_set_q1_1), len(answer_set_q1_2))
        answers_00 = answer_set_q1_1 & answer_set_q1_2
        for ele in answers_00:
            answers_101.update(test_only_ent_out[ele][rel3])

        answers_011 = set()
        answer_set_q1_1 = train_ent_out[entity1][rel1]
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        n_intermediate_ext_answers = max(len(answer_set_q1_1), len(answer_set_q1_2))
        answers_00 = answer_set_q1_1 & answer_set_q1_2
        for ele in answers_00:
            answers_011.update(test_only_ent_out[ele][rel3])

        reachable_answers_1p = (answers_001 | answers_101 | answers_011) - reachable_answers_0p - train_answer_set
        # reachable_answers_1p = answers_001 - reachable_answers_0p - train_answer_set
        # reachable_answers_1p = (answers_101 | answers_011) - reachable_answers_0p - train_answer_set

        answers_110 = set()
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        n_intermediate_ext_answers = max(len(answer_set_q1_1), len(answer_set_q1_2))
        answers_11 = answer_set_q1_1 & answer_set_q1_2
        for ele in answers_11:
            answers_110.update(train_ent_out[ele][rel3])
        reachable_answers_2u = answers_110 - reachable_answers_1p - reachable_answers_0p - train_answer_set

        # 111 #up reduced
        answers_111 = set()
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        n_intermediate_ext_answers = max(len(answer_set_q1_1), len(answer_set_q1_2))
        answers_11 = answer_set_q1_1 & answer_set_q1_2
        for ele in answers_11:
            answers_111.update(test_only_ent_out[ele][rel3])
        reachable_answers_up = answers_111 - reachable_answers_1p - reachable_answers_2u - reachable_answers_0p - train_answer_set

        if len(reachable_answers_up) > 0:
            if rel1 in rel_per_query:
                rel_per_query[rel1] += len(reachable_answers_up)
            else:
                rel_per_query[rel1] = len(reachable_answers_up)
            if rel1 != rel2:
                if rel2 in rel_per_query:
                    rel_per_query[rel2] += len(reachable_answers_up)
                else:
                    rel_per_query[rel2] = len(reachable_answers_up)
            if rel1 != rel3 and rel2 != rel3:
                if rel3 in rel_per_query:
                    rel_per_query[rel3] += len(reachable_answers_up)
                else:
                    rel_per_query[rel3] = len(reachable_answers_up)

            if entity1 in anch_per_query:
                anch_per_query[entity1] += len(reachable_answers_up)
            else:
                anch_per_query[entity1] = len(reachable_answers_up)
            if entity1 != entity2:
                if entity2 in anch_per_query:
                    anch_per_query[entity2] += len(reachable_answers_up)
                else:
                    anch_per_query[entity2] = len(reachable_answers_up)
        return reachable_answers_up, rel_per_query, anch_per_query, [rel1, rel2, rel3], [entity1, entity2]

    # 2u
    if query_structure == [['e', ['r']], ['e', ['r']], ['u']]:
        entity1 = query[0][0]
        rel1 = query[0][1][0]
        entity2 = query[1][0]
        rel2 = query[1][1][0]
        reachable_answers_2u = set()
        # 01
        # compute the answers of the query (entity1,rel1,?y) on the training graph

        answer_set_q1_1 = train_ent_out[entity1][rel1]
        # compute the answers of the query (entity2,rel2,?y) on the missing graph
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        answers_01 = answer_set_q1_1 & answer_set_q1_2
        # 10
        # compute the answers of the query (entity1,rel1,?y) on the missing graph
        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        # compute the answers of the query (entity2,rel2,?y) on the training graph
        answer_set_q1_2 = train_ent_out[entity2][rel2]
        answers_10 = answer_set_q1_1 & answer_set_q1_2


        reachable_answers_1p = (answers_01 | answers_10) - train_answer_set

        answer_set_q1_1 = test_only_ent_out[entity1][rel1]
        # compute the answers of the query (entity2,rel2,?y) on the training graph
        answer_set_q1_2 = test_only_ent_out[entity2][rel2]
        answers_11 = answer_set_q1_1 & answer_set_q1_2
        reachable_answers_2u = answers_11 - reachable_answers_1p - train_answer_set
        if len(reachable_answers_2u) > 0:
            if rel1 in rel_per_query:
                rel_per_query[rel1] += len(reachable_answers_2u)
            else:
                rel_per_query[rel1] = len(reachable_answers_2u)
            if rel1 != rel2:
                if rel2 in rel_per_query:
                    rel_per_query[rel2] += len(reachable_answers_2u)
                else:
                    rel_per_query[rel2] = len(reachable_answers_2u)

            if entity1 in anch_per_query:
                anch_per_query[entity1] += len(reachable_answers_2u)
            else:
                anch_per_query[entity1] = len(reachable_answers_2u)
            if entity1 != entity2:
                if entity2 in anch_per_query:
                    anch_per_query[entity2] += len(reachable_answers_2u)
                else:
                    anch_per_query[entity2] = len(reachable_answers_2u)
        return reachable_answers_2u, rel_per_query, anch_per_query, [rel1, rel2], [entity1, entity2]


def top_k_dict_values_sorting(d, k):
    # Sort the dictionary by values in descending order and get the top-k items
    sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=True)[:k]
    # Extract the keys and values from the sorted items
    top_k_keys = [item[0] for item in sorted_items]
    top_k_values = [item[1] for item in sorted_items]
    return top_k_keys, top_k_values

def get_top_k_frequency(dataset, num_sampled,ele_per_query, type):
    top_k_keys, top_k_values = top_k_dict_values_sorting(ele_per_query, 5)
    #print(top_k_values)
    perc_top = (top_k_values[0] * 100) / num_sampled
    name_top = top_k_keys[0]
    if type == 'rel':
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
    return data_dict[name_top], perc_top

def ground_queries(dataset, query_structures, ent_in, ent_out, train_ent_in, train_ent_out, test_only_ent_in,
                   test_only_ent_out, gen_num, max_ans_num, query_names, mode, ent2id, rel2id):
    #print(mode)
    queries = defaultdict(set)
    train_answers = defaultdict(set)
    hard_answers = defaultdict(set)
    rel_per_query_overall = defaultdict()
    anch_per_query_overall = defaultdict()
    s0 = time.time()
    random.seed(0)
    perc_top_rel = 0
    perc_top_anch = 0
    for idx, query_structure in enumerate(query_structures):
        query_name = query_names[idx]
        rel_per_query_overall[list2tuple(query_structure)] = {}
        anch_per_query_overall[list2tuple(query_structure)] = {}
        print('general structure is', query_structure, "with name", query_name)
        ## instead on checking for the gen_num/num_sampled, check the number of answers that figure as non-reduceable; we want 50.000 per query structure
        num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
        train_ans_num, hard_ans_num = [], []
        old_num_sampled = -1
        while num_sampled < gen_num:
            if num_sampled != 0:
                if num_sampled % (gen_num // 100) == 0 and num_sampled != old_num_sampled:
                    logging.info(
                        '%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s' % (
                        mode,
                        query_structure,
                        num_sampled, gen_num, (time.time() - s0) / num_sampled, num_try, num_repeat, num_more_answer,
                        num_broken, num_no_extra_answer, num_no_extra_negative, num_empty))
                    old_num_sampled = num_sampled
            # print ('%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s'%(mode,
            #    query_structure,
            #    num_sampled, gen_num, (time.time()-s0)/(num_sampled+0.001), num_try, num_repeat, num_more_answer,
            #    num_broken, num_no_extra_answer, num_no_extra_negative, num_empty), end='\r')
            num_try += 1
            empty_query_structure = deepcopy(query_structure)
            # answer = random.sample(ent_in.keys(), 1)[0]
            answer = random.sample(test_only_ent_in.keys(), 1)[
                0]  # we search only in the set of entities that appear in the test
            broken_flag = fill_query(empty_query_structure, ent_in, ent_out, answer, ent2id, rel2id)
            if broken_flag:
                num_broken += 1
                continue
            query = empty_query_structure
            answer_set = achieve_answer(query, ent_in, ent_out)
            train_answer_set = achieve_answer(query, train_ent_in, train_ent_out)  # only training
            test_answer_set = answer_set - train_answer_set
            if list2tuple(query) in queries[list2tuple(query_structure)]:
                num_repeat += 1
                continue
            if len(answer_set) == 0:
                num_empty += 1
                continue
            if mode != 'train':
                if len(answer_set - train_answer_set) == 0:
                    num_no_extra_answer += 1
                    continue
                if 'n' in query_name:
                    if len(train_answer_set - answer_set) == 0:
                        num_no_extra_negative += 1
                        continue
            # nr_answers,rel_per_query,anch_per_query,rels,anchs = compute_nr_answers(query_structure, query,test_only_ent_out, train_ent_out, train_answer_set, test_answer_set)
            nr_answers, rel_per_query_temp, anch_per_query_temp, rels, anchs = compute_nr_answers(query_structure,
                                                                                                  query,
                                                                                                  test_only_ent_out,
                                                                                                  train_ent_out,
                                                                                                  train_answer_set,
                                                                                                  test_answer_set,
                                                                                                  rel_per_query_overall[
                                                                                                      list2tuple(
                                                                                                          query_structure)].copy(),
                                                                                                  anch_per_query_overall[
                                                                                                      list2tuple(
                                                                                                          query_structure)].copy())
            if len(nr_answers) == 0:
                num_no_extra_answer += 1
                continue

            if max(len(answer_set - train_answer_set), len(train_answer_set - answer_set)) > max_ans_num:
                num_more_answer += 1
                continue

            # rel_per_query_temp = {key: rel_per_query_overall[list2tuple(query_structure)].get(key, 0) + rel_per_query.get(key, 0) for key in
            #                      set(rel_per_query_overall[list2tuple(query_structure)]) | set(rel_per_query)}
            # anch_per_query_temp = {
            #    key: anch_per_query_overall[list2tuple(query_structure)].get(key, 0) + anch_per_query.get(key, 0) for key
            #    in
            #    set(anch_per_query_overall[list2tuple(query_structure)]) | set(anch_per_query)}
            if num_sampled > gen_num / 4:
                top_k_rel_keys, top_k_rel_values = top_k_dict_values_sorting(rel_per_query_temp, 1)
                perc_top_rel = (top_k_rel_values[0] * 100) / (num_sampled + len(nr_answers))

                top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_temp, 1)
                perc_top_anch = (top_k_anch_values[0] * 100) / (num_sampled + len(nr_answers))

                if (perc_top_rel > 20 and top_k_rel_keys[0] in rels) or (
                        perc_top_anch > 20 and top_k_anch_keys[0] in anchs):
                    num_more_answer += 1
                    continue

            rel_per_query_overall[list2tuple(query_structure)] = rel_per_query_temp
            anch_per_query_overall[list2tuple(query_structure)] = anch_per_query_temp

            queries[list2tuple(query_structure)].add(list2tuple(query))
            train_answers[list2tuple(query)] = answer_set - nr_answers  # train answer set
            hard_answers[list2tuple(query)] = nr_answers  # hard answer set
            num_sampled += len(nr_answers)
            # num_answer_generated += n_non_reduceable_answers
            # store somewhere also the number of reduceable (?)
            train_ans_num.append(len(train_answers[list2tuple(query)]))
            hard_ans_num.append(len(hard_answers[list2tuple(query)]))
        #logging.info(perc_top_rel)
        #logging.info(perc_top_anch)
        key_rel, perc_rel = get_top_k_frequency(dataset, num_sampled,rel_per_query_overall[list2tuple(query_structure)], 'rel')
        key_ent, perc_anch = get_top_k_frequency(dataset, num_sampled, anch_per_query_overall[list2tuple(query_structure)], 'ent')
        logging.info(num_sampled)
        logging.info("Most present relation name: " + str(key_rel))
        logging.info("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_rel))
        logging.info("Most present anchor: " + str(key_ent))
        logging.info("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_anch))

    print()
    #logging.info("{} tp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(train_ans_num), np.min(train_ans_num),
    #                                                                np.mean(train_ans_num), np.std(train_ans_num)))
    #logging.info("{} fn max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(hard_ans_num), np.min(hard_ans_num),
    #                                                                np.mean(hard_ans_num), np.std(hard_ans_num)))
    #logging.info("num sampled answers: {}".format(num_sampled))

    name_to_save = '%s-' % (mode)
    with open('./data/%s/%s-adj-new-queries.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-adj-new-easy-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(train_answers, f)
    with open('./data/%s/%s-adj-new-hard-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(hard_answers, f)
    return queries, train_answers, hard_answers


def generate_queries(dataset, query_structures, gen_num, max_ans_num, gen_train, gen_valid, gen_test, query_names,
                     save_name):
    base_path = './data/%s' % dataset
    indexified_files = ['train.txt', 'valid.txt', 'test.txt']
    if gen_train or gen_valid:
        train_ent_in, train_ent_out = construct_graph(base_path, indexified_files[:1])  # ent_in 
    if gen_valid or gen_test:
        valid_ent_in, valid_ent_out = construct_graph(base_path, indexified_files[:2])
        valid_only_ent_in, valid_only_ent_out = construct_graph(base_path, indexified_files[1:2])
    if gen_test:
        test_ent_in, test_ent_out = construct_graph(base_path, indexified_files[:3])
        test_only_ent_in, test_only_ent_out = construct_graph(base_path, indexified_files[2:3])

    ent2id = pickle.load(open(os.path.join(base_path, "ent2id.pkl"), 'rb'))
    rel2id = pickle.load(open(os.path.join(base_path, "rel2id.pkl"), 'rb'))

    train_queries = defaultdict(set)
    train_train_answers = defaultdict(set)
    train_fp_answers = defaultdict(set)
    train_hard_answers = defaultdict(set)
    valid_queries = defaultdict(set)
    valid_train_answers = defaultdict(set)
    valid_fp_answers = defaultdict(set)
    valid_hard_answers = defaultdict(set)
    test_queries = defaultdict(set)
    test_answers = defaultdict(set)
    test_train_answers = defaultdict(set)
    test_fp_answers = defaultdict(set)
    test_hard_answers = defaultdict(set)

    t1, t2, t3, t4, t5, t6 = 0, 0, 0, 0, 0, 0
    # assert len(query_structures) == 1

    # print ('general structure is', query_structure, "with name", query_name)
    # if query_structure == ['e', ['r']]:
    #    if gen_train:
    #        write_links(dataset, train_ent_out, defaultdict(lambda: defaultdict(set)), max_ans_num, 'train-'+query_name)
    #    if gen_valid:
    #        write_links(dataset, valid_only_ent_out, train_ent_out, max_ans_num, 'valid-'+query_name)
    #    if gen_test:
    #        write_links(dataset, test_only_ent_out, valid_ent_out, max_ans_num, 'test-'+query_name)
    #    print ("link prediction created!")
    #    exit(-1)

    formatted_date_time = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    name_to_save = formatted_date_time + "_new_bench"
    set_logger("./data/{}/".format(dataset), name_to_save)
    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_empty = 0, 0, 0, 0, 0, 0
    train_ans_num = []
    s0 = time.time()
    if gen_train:
        train_queries, train_train_answers, train_hard_answers = ground_queries(dataset, query_structures,
                                                                                train_ent_in, train_ent_out,
                                                                                defaultdict(lambda: defaultdict(set)),
                                                                                defaultdict(lambda: defaultdict(set)),
                                                                                None, None,
                                                                                gen_num[0], max_ans_num, query_names,
                                                                                'train', ent2id, rel2id)
    if gen_valid:
        valid_queries, valid_train_answers, valid_hard_answers = ground_queries(dataset, query_structures,
                                                                                valid_ent_in, valid_ent_out,
                                                                                train_ent_in, train_ent_out,
                                                                                valid_only_ent_in, valid_only_ent_out,
                                                                                gen_num[1], max_ans_num, query_names,
                                                                                'valid', ent2id, rel2id)
    if gen_test:
        test_queries, test_train_answers, test_hard_answers = ground_queries(dataset, query_structures,
                                                                             test_ent_in, test_ent_out, valid_ent_in,
                                                                             valid_ent_out, test_only_ent_in,
                                                                             test_only_ent_out, gen_num[2], max_ans_num,
                                                                             query_names, 'test', ent2id, rel2id)
    print(dataset)
    print('%s queries generated with structure %s' % (gen_num, query_structures))


def fill_query(query_structure, ent_in, ent_out, answer, ent2id, rel2id):
    assert type(query_structure[-1]) == list
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        r = -1
        for i in range(len(query_structure[-1]))[::-1]:
            if query_structure[-1][i] == 'n':
                query_structure[-1][i] = -2
                continue
            found = False
            for j in range(40):
                r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                if r_tmp // 2 != r // 2 or r_tmp == r:
                    r = r_tmp
                    found = True
                    break
            if not found:
                return True
            query_structure[-1][i] = r
            answer = random.sample(ent_in[answer][r], 1)[0]
        if query_structure[0] == 'e':
            query_structure[0] = answer
        else:
            return fill_query(query_structure[0], ent_in, ent_out, answer, ent2id, rel2id)
    else:
        same_structure = defaultdict(list)
        for i in range(len(query_structure)):
            same_structure[list2tuple(query_structure[i])].append(i)
        for i in range(len(query_structure)):
            if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                assert i == len(query_structure) - 1
                query_structure[i][0] = -1
                continue
            broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, ent2id, rel2id)
            if broken_flag:
                return True
        for structure in same_structure:
            if len(same_structure[structure]) != 1:
                structure_set = set()
                for i in same_structure[structure]:
                    structure_set.add(list2tuple(query_structure[i]))
                if len(structure_set) < len(same_structure[structure]):
                    return True


def achieve_answer(query, ent_in, ent_out):
    assert type(query[-1]) == list
    all_relation_flag = True
    for ele in query[-1]:
        if (type(ele) != int) or (ele == -1):
            all_relation_flag = False
            break
    if all_relation_flag:
        if type(query[0]) == int:
            ent_set = set([query[0]])
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out)
        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                ent_set = set(range(len(ent_in))) - ent_set
            else:
                ent_set_traverse = set()
                for ent in ent_set:
                    ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                ent_set = ent_set_traverse
    else:
        ent_set = achieve_answer(query[0], ent_in, ent_out)
        union_flag = False
        if len(query[-1]) == 1 and query[-1][0] == -1:
            union_flag = True
        for i in range(1, len(query)):
            if not union_flag:
                ent_set = ent_set.intersection(achieve_answer(query[i], ent_in, ent_out))
            else:
                if i == len(query) - 1:
                    continue
                ent_set = ent_set.union(achieve_answer(query[i], ent_in, ent_out))
    return ent_set


@click.command()
@click.option('--dataset', default="FB15k-237-betae")
@click.option('--seed', default=0)
@click.option('--gen_train_num', default=0)
@click.option('--gen_valid_num', default=0)
@click.option('--gen_test_num', default=0)
@click.option('--max_ans_num', default=1e6)
@click.option('--reindex', is_flag=True, default=False)
@click.option('--gen_train', is_flag=True, default=False)
@click.option('--gen_valid', is_flag=True, default=False)
@click.option('--gen_test', is_flag=True, default=False)
@click.option('--gen_id', default=0)
@click.option('--save_name', is_flag=True, default=False)
@click.option('--index_only', is_flag=True, default=False)
@click.option('--fourp', is_flag=True, default=False)
def main(dataset, seed, gen_train_num, gen_valid_num, gen_test_num, max_ans_num, reindex, gen_train, gen_valid,
         gen_test, gen_id, save_name, index_only, fourp):
    '''
    train_num_dict = {'FB15k': 273710, "FB15k-237": 149689, "NELL": 107982}
    valid_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000}
    test_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000}
    if gen_train and gen_train_num == 0:
        if 'FB15k-237' in dataset:
            gen_train_num = 149689
        elif 'FB15k' in dataset:
            gen_train_num = 273710
        elif 'NELL' in dataset:
            gen_train_num = 107982
        else:
            gen_train_num = train_num_dict[dataset]
    if gen_valid and gen_valid_num == 0:
        if 'FB15k-237' in dataset:
            gen_valid_num = 5000
        elif 'FB15k' in dataset:
            gen_valid_num = 8000
        elif 'NELL' in dataset:
            gen_valid_num = 4000
        else:
            gen_valid_num = valid_num_dict[dataset]
    if gen_test and gen_test_num == 0:
        if 'FB15k-237' in dataset:
            gen_test_num = 5000
        elif 'FB15k' in dataset:
            gen_test_num = 8000
        elif 'NELL' in dataset:
            gen_test_num = 4000
        else:
            gen_test_num = test_num_dict[dataset]
    if index_only:
        index_dataset(dataset, reindex)
        exit(-1)
    '''
    if index_only:
        index_dataset(dataset, reindex)
        exit(-1)
    e = 'e'
    r = 'r'
    n = 'n'
    u = 'u'
    query_structures = [
        # [e, [r]],
        #[e, [r, r]],
        #[e, [r, r, r]],
        #[[e, [r]], [e, [r]]],
        #[[e, [r]], [e, [r]], [e, [r]]],
        #[[e, [r, r]], [e, [r]]],
        #[[[e, [r]], [e, [r]]], [r]],
        # negation
        # [[e, [r]], [e, [r, n]]],
        # [[e, [r]], [e, [r]], [e, [r, n]]],
        # [[e, [r, r]], [e, [r, n]]],
        # [[e, [r, r, n]], [e, [r]]],
        # [[[e, [r]], [e, [r, n]]], [r]],
        # union
        #[[e, [r]], [e, [r]], [u]],
        #[[[e, [r]], [e, [r]], [u]], [r]],
        # harder?
        [e, [r, r, r, r]],
        [[e, [r]], [e, [r]], [e, [r]], [e, [r]]],
    ]
    # query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']
    #query_names = ['2p', '3p', '2i', '3i', 'pi', 'ip', '2u', 'up']#, '4p', '4i']
    query_names = ['4p', '4i']
    #query_names = ['pi', 'ip', '2u', 'up']  #
    # print(query_structures)
    # print(dataset)
    gen_test_num = gen_valid_num = 50000
    generate_queries(dataset, query_structures, [gen_train_num, gen_valid_num, gen_test_num], max_ans_num, gen_train,
                     gen_valid, gen_test, query_names, save_name)


if __name__ == '__main__':
    main()