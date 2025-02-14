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


def write_links(dataset, eval_ent_out, train_ent_out, max_ans_num, name):
    queries = defaultdict(set)
    train_answers = defaultdict(set)
    hard_answers = defaultdict(set)
    fp_answers = defaultdict(set)
    num_more_answer = 0
    for ent in eval_ent_out:
        for rel in eval_ent_out[ent]:
            if len(eval_ent_out[ent][rel]) <= max_ans_num:
                queries[('e', ('r',))].add((ent, (rel,)))
                train_answers[(ent, (rel,))] = train_ent_out[ent][rel]
                hard_answers[(ent, (rel,))] = eval_ent_out[ent][rel]
            else:
                num_more_answer += 1
    if name == 'train-':
        with open('./data/%s/%s-1p-queries.pkl' % (dataset, name), 'wb') as f:
            pickle.dump(queries, f)
        with open('./data/%s/%s-1p-answers.pkl' % (dataset, name), 'wb') as f:
            pickle.dump(hard_answers, f)
    else:
        with open('./data/%s/%s-1p-queries.pkl' % (dataset, name), 'wb') as f:
            pickle.dump(queries, f)
        with open('./data/%s/%seasy-1p-answers.pkl' % (dataset, name), 'wb') as f:
            pickle.dump(train_answers, f)
        with open('./data/%s/%shard-1p-answers.pkl' % (dataset, name), 'wb') as f:
            pickle.dump(hard_answers, f)
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


def add_to_freq_dict(n_answers, set_rels,set_anchs,rels_freq,anch_freqs):
    new_set_rel = set()
    for rel in set_rels:
        if rel % 2 != 0:
            # rel is the inverse
            rel = rel - 1
        new_set_rel.update({rel})
    for rel in new_set_rel:
        if rel in rels_freq:
            rels_freq[rel] += n_answers
        else:
            rels_freq[rel] = n_answers

    for entity in set_anchs:
        if entity in anch_freqs:
            anch_freqs[entity] += n_answers
        else:
            anch_freqs[entity] = n_answers
    return rels_freq, anch_freqs, new_set_rel


def compute_fi_pi_answers(query_structure, query, test_only_ent_out, train_ent_out, all_ent_out, train_answer_set, test_answer_set, answer_set):
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
        return len(reachable_answers_2p),reachable_answers_2p, [reachable_answers_1p],[rel1, rel2], [entity]
    # 3p
    if query_structure == ['e', ['r', 'r', 'r']]:
        reachable_answers_3p = set()

        entity = query[0]
        rel1 = query[1][0]
        rel2 = query[1][1]
        rel3 = query[1][2]
        reachable_answers_2p = set()
        # existing + existing + predicted --> 1p 001
        reachable_answers_001 = compute_answers_query_3p(entity, [rel1, rel2, rel3], train_ent_out, train_ent_out,
                                                         test_only_ent_out)
        # existing + predicted + existing --> 1p 010
        reachable_answers_010 = compute_answers_query_3p(entity, [rel1, rel2, rel3], train_ent_out, test_only_ent_out,
                                                         train_ent_out)
        # predicted + existing + existing --> 1p 100
        reachable_answers_100 = compute_answers_query_3p(entity, [rel1, rel2, rel3], test_only_ent_out, train_ent_out,
                                                         train_ent_out)
        reachable_answers_1p = (reachable_answers_001 | reachable_answers_010 | reachable_answers_100) - train_answer_set  # subtract the train answers

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
            reachable_answers_2p = (reachable_answers_011 | reachable_answers_110 | reachable_answers_101) - reachable_answers_1p - train_answer_set  # subtract the train answers and the 1p answers
            if len(reachable_answers_1p | reachable_answers_2p) < len(test_answer_set):
                # predicted + predicted + existing --> 3p 111
                reachable_answers_111 = compute_answers_query_3p(entity, [rel1, rel2, rel3], test_only_ent_out,
                                                                 test_only_ent_out, test_only_ent_out)
                reachable_answers_3p = reachable_answers_111 - reachable_answers_2p - reachable_answers_1p - train_answer_set  # subtract the train answers and the 1p/2p answers

        return len(reachable_answers_3p),reachable_answers_3p, [reachable_answers_1p,reachable_answers_2p],[rel1, rel2, rel3], [entity]

    # 4p
    if query_structure == ['e', ['r', 'r', 'r', 'r']]:

        reachable_answers_4p = set()
        entity = query[0]
        rel1 = query[1][0]
        rel2 = query[1][1]
        rel3 = query[1][2]
        rel4 = query[1][3]
        reachable_answers_2p, reachable_answers_3p = set(), set()
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

        return len(reachable_answers_4p),reachable_answers_4p, [reachable_answers_1p,reachable_answers_2p,reachable_answers_3p], [rel1, rel2, rel3, rel4], [entity]

    # 2i
    if query_structure == [['e', ['r']], ['e', ['r']]]:
        reachable_answers_2i = set()
        # 2i
        entity1 = query[0][0]
        rel1 = query[0][1][0]
        entity2 = query[1][0]
        rel2 = query[1][1][0]
        set_rel = ({rel1, rel2})
        set_ent = ({entity1, entity2})
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

        return len(reachable_answers_2i),reachable_answers_2i, [reachable_answers_1p], [rel1, rel2], [entity1, entity2]
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
        set_rel = ({rel1, rel2, rel3})
        set_ent = ({entity1, entity2, entity3})
        reachable_answers_2i = set()
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

        return len(reachable_answers_3i), reachable_answers_3i, [
            reachable_answers_1p,reachable_answers_2i ], [
                   rel1, rel2,rel3], [entity1, entity2, entity3]

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
        set_rel = ({rel1, rel2, rel3, rel4})
        set_ent = ({entity1, entity2, entity3, entity4})
        reachable_answers_2i = set()
        reachable_answers_3i = set()
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
                if len(reachable_answers_3i)>0:
                    print()

                if len(reachable_answers_3i |reachable_answers_2i | reachable_answers_1p) < len(test_answer_set):
                    # 1111
                    answer_set_q1_1 = test_only_ent_out[entity1][rel1]
                    answer_set_q1_2 = test_only_ent_out[entity2][rel2]
                    answer_set_q1_3 = test_only_ent_out[entity3][rel3]
                    answer_set_q1_4 = test_only_ent_out[entity4][rel4]
                    answers_1111 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                    reachable_answers_4i = answers_1111 - reachable_answers_1p - reachable_answers_2i- reachable_answers_3i - train_answer_set

        return len(reachable_answers_4i), reachable_answers_4i, [reachable_answers_1p,reachable_answers_2i,reachable_answers_3i], [
                   rel1, rel2,rel3,rel4], [entity1, entity2,entity3,entity4]
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
        set_rel = ({rel1, rel2, rel3})
        set_ent = ({entity1, entity2})
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

        return len(reachable_answers_pi),reachable_answers_pi, [reachable_answers_1p,reachable_answers_2i, reachable_answers_2p], [rel1, rel2, rel3], [entity1, entity2]
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
        set_rel = ({rel1, rel2, rel3})
        set_ent = ({entity1, entity2})
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

        return len(reachable_answers_ip), reachable_answers_ip, [reachable_answers_1p, reachable_answers_2i,
                                                                 reachable_answers_2p], [
                   rel1, rel2, rel3], [entity1, entity2]

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

        return len(reachable_answers_up), reachable_answers_up, [reachable_answers_1p, reachable_answers_2u], [
                   rel1, rel2, rel3], [entity1, entity2]
    # 2u
    if query_structure == [['e', ['r']], ['e', ['r']], ['u']]:
        entity1 = query[0][0]
        rel1 = query[0][1][0]
        entity2 = query[1][0]
        rel2 = query[1][1][0]
        reachable_answers_2u = set()
        rel_per_query_partial_inf = {}
        anch_per_query_partial_inf = {}
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
        return len(reachable_answers_2u),reachable_answers_2u, [],[rel1, rel2], [entity1, entity2]

    # Negation
    # 2in
    if query_structure == [['e', ['r']], ['e', ['r', 'n']]]:
        # n_tot_hard_answers_2in += len(hard_answer_set)
        entity1 = query[0][0]
        rel1 = query[0][1][0]  # positive
        entity2 = query[1][0]
        rel2 = query[1][1][0]  # negative
        all_ent_r2 = all_ent_out[entity2][rel2]
        train_ent_r1 = train_ent_out[entity1][rel1]
        train_ent_r2 = train_ent_out[entity2][rel2]
        missing_ent_r1 = test_only_ent_out[entity1][rel1]

        # easy answer set
        new_train_answer_set = (train_ent_r1 - train_ent_r2) & answer_set
        answers0x = train_ent_r1 - all_ent_r2
        answers1x = missing_ent_r1 - all_ent_r2
        reachable_2in_pos_exist = answers0x - new_train_answer_set
        reachable_2in_pos_only_missing = answers1x - reachable_2in_pos_exist - new_train_answer_set
        return len(reachable_2in_pos_only_missing),reachable_2in_pos_only_missing, [reachable_2in_pos_exist],[rel1, rel2], [entity1, entity2]

    # 3in
    if query_structure == [['e', ['r']], ['e', ['r']], ['e', ['r', 'n']]]:
        entity1 = query[0][0]
        rel1 = query[0][1][0]  # positive
        entity2 = query[1][0]
        rel2 = query[1][1][0]  # positive
        entity3 = query[2][0]
        rel3 = query[2][1][0]  # negative

        all_ent_r3 = all_ent_out[entity3][rel3]
        train_ent_r1 = train_ent_out[entity1][rel1]
        train_ent_r2 = train_ent_out[entity2][rel2]
        train_ent_r3 = train_ent_out[entity3][rel3]
        missing_ent_r1 = test_only_ent_out[entity1][rel1]
        missing_ent_r2 = test_only_ent_out[entity2][rel2]
        missing_ent_r3 = test_only_ent_out[entity3][rel3]

        # easy answer set
        new_train_answer_set = train_answer_set & answer_set
        new_hard_answer_set = answer_set - new_train_answer_set
        # answers00x = (train_ent_r1 & train_ent_r2) - all_ent_r3
        answers01x = (train_ent_r1 & missing_ent_r2) - all_ent_r3
        answers10x = (missing_ent_r1 & train_ent_r2) - all_ent_r3
        answers11x = (missing_ent_r1 & missing_ent_r2) - all_ent_r3
        reachable_3in_pos_exist = (answers01x | answers10x) - new_train_answer_set
        reachable_3in_pos_only_missing = answers11x - reachable_3in_pos_exist - new_train_answer_set

        return len(reachable_3in_pos_only_missing), reachable_3in_pos_only_missing, [reachable_3in_pos_exist], [
            rel1, rel2, rel3], [entity1, entity2, entity3]

    # pin
    if query_structure == [['e', ['r', 'r']], ['e', ['r', 'n']]]:
        # n_tot_hard_answers_pin += len(hard_answer_set)
        entity1 = query[0][0]
        rel1 = query[0][1][0]
        rel2 = query[0][1][1]
        entity2 = query[1][0]
        rel3 = query[1][1][0]  # negation

        all_ent_r3 = all_ent_out[entity2][rel3]

        trainr1_trainr2 = compute_answers_query_2p(entity1, [rel1, rel2], train_ent_out, train_ent_out)
        trainr1_missingr2 = compute_answers_query_2p(entity1, [rel1, rel2], train_ent_out, test_only_ent_out)
        missingr1_trainr2 = compute_answers_query_2p(entity1, [rel1, rel2], test_only_ent_out, train_ent_out)
        missingr1_missingr2 = compute_answers_query_2p(entity1, [rel1, rel2], test_only_ent_out, test_only_ent_out)

        # easy answer set
        new_train_answer_set = train_answer_set & answer_set
        new_hard_answer_set = answer_set - new_train_answer_set
        # answers00x = trainr1_trainr2 - all_ent_r3
        answers01x = trainr1_missingr2 - all_ent_r3
        answers10x = missingr1_trainr2 - all_ent_r3
        answers11x = missingr1_missingr2 - all_ent_r3
        reachable_pin_pos_exist = (answers01x | answers10x) - new_train_answer_set
        reachable_pin_pos_only_missing = answers11x - reachable_pin_pos_exist - new_train_answer_set
        return len(reachable_pin_pos_only_missing), reachable_pin_pos_only_missing, [reachable_pin_pos_exist], [
            rel1, rel2, rel3], [entity1, entity2]

    # pni
    if query_structure == [['e', ['r', 'r', 'n']], ['e', ['r']]]:
        entity1 = query[0][0]
        rel1 = query[0][1][0]
        rel2 = query[0][1][1]  # negation on the 2p path query
        entity2 = query[1][0]
        rel3 = query[1][1][0]
        # easy answer set
        new_train_answer_set = train_answer_set & answer_set
        all_answers_2p = compute_answers_query_2p(entity1, [rel1, rel2], all_ent_out, all_ent_out)
        easy_ent_r3 = train_ent_out[entity2][rel3]
        missing_ent_r3 = test_only_ent_out[entity2][rel3]
        answersxx0 = easy_ent_r3 - all_answers_2p
        answersxx1 = missing_ent_r3 - all_answers_2p
        reachable_pni_pos_exist = answersxx0 - new_train_answer_set
        reachable_pni_pos_only_missing = answersxx1 - new_train_answer_set
        return len(reachable_pni_pos_only_missing), reachable_pni_pos_only_missing, [reachable_pni_pos_exist], [
            rel1, rel2, rel3], [entity1, entity2]

    # inp
    if query_structure == [[['e', ['r']], ['e', ['r', 'n']]], ['r']]:
        entity1 = query[0][0][0]
        rel1 = query[0][0][1][0]
        entity2 = query[0][1][0]
        rel2 = query[0][1][1][0]  # negated
        rel3 = query[1][0]

        all_ent_r2 = all_ent_out[entity2][rel2]
        train_ent_r1 = train_ent_out[entity1][rel1]
        missing_ent_r1 = test_only_ent_out[entity1][rel1]

        # easy answer set
        new_train_answer_set = train_answer_set & answer_set
        new_hard_answer_set = answer_set - new_train_answer_set

        answers_0x0 = set()
        answers_0x1 = set()
        answers_1x0 = set()
        answers_1x1 = set()
        answers_0x = train_ent_r1 - all_ent_r2
        for ele in answers_0x:
            answers_0x0.update(train_ent_out[ele][rel3])
        for ele in answers_0x:
            answers_0x1.update(test_only_ent_out[ele][rel3])
        answers_1x = missing_ent_r1 - all_ent_r2
        for ele in answers_1x:
            answers_1x0.update(train_ent_out[ele][rel3])
        for ele in answers_1x:
            answers_1x1.update(test_only_ent_out[ele][rel3])

        reachable_inp_pos_exist = (answers_0x1 | answers_1x0) - new_train_answer_set
        reachable_inp_pos_only_missing = answers_1x1 - reachable_inp_pos_exist - new_train_answer_set
        return len(reachable_inp_pos_only_missing), reachable_inp_pos_only_missing, [reachable_inp_pos_exist], [
            rel1, rel2, rel3], [entity1, entity2]

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

def ground_queries(dataset, query_structures, all_ent_in, all_ent_out, train_ent_in, train_ent_out, test_only_ent_in,
                   test_only_ent_out, gen_num, gen_num_per_query, max_ans_num, query_names, mode, ent2id, rel2id,seed):
    queries = defaultdict(set)
    filtered_answers = defaultdict(set)
    hard_answers = defaultdict(set)
    rel_per_query_overall = defaultdict()
    anch_per_query_overall = defaultdict()
    s0 = time.time()
    random.seed(seed)
    gen_num_origin = gen_num


    for idx, query_structure in enumerate(query_structures):
        query_name = query_names[idx]
        gen_num = gen_num_origin
        rel_per_query_overall[list2tuple(query_structure)] = {}
        anch_per_query_overall[list2tuple(query_structure)] = {}
        print('general structure is', query_structure, "with name", query_name)
        ## instead on checking for the gen_num/num_sampled, check the number of answers that figure as non-reduceable; we want 50.000 per query structure
        num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
        tot_qa_pairs = 0
        train_ans_num, hard_ans_num = [], []
        old_num_sampled = -1
        gen_num_partials = gen_num_per_query[query_name]
        tot_to_gen = gen_num * (len(gen_num_partials) + 1)
        while tot_qa_pairs < tot_to_gen:
            if num_sampled != 0:
                if num_sampled % (gen_num // 10) == 0 and num_sampled != old_num_sampled:
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
            answer = random.sample(test_only_ent_in.keys(), 1)[0]  # we search only in the set of entities that appear in the test
            broken_flag = fill_query(empty_query_structure, all_ent_in, all_ent_out, answer, ent2id, rel2id)
            if broken_flag:
                num_broken += 1
                continue
            query = empty_query_structure
            answer_set = achieve_answer(query, all_ent_in, all_ent_out)
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
            n_sampled_step, full_inf_answers, partial_inf_answers,rels, anchs = compute_fi_pi_answers(
                query_structure,
                query,
                test_only_ent_out,
                train_ent_out, all_ent_out,
                train_answer_set,
                test_answer_set, answer_set)
            if num_sampled < gen_num:
                if n_sampled_step == 0: #check at least one full-inf answer
                    num_no_extra_answer += 1
                    continue
            if num_sampled + n_sampled_step <= gen_num:
                answerstoadd = full_inf_answers
            else:
                k_to_add = gen_num - num_sampled
                n_sampled_step = k_to_add
                answerstoadd = set(random.sample(full_inf_answers, k_to_add))

            copy_gen_num_partials = gen_num_partials.copy()
            for gnidx, gen_num_partial in enumerate(copy_gen_num_partials):
                if gen_num_partial < gen_num:
                    temp_tot_partials = copy_gen_num_partials[gnidx]+len(partial_inf_answers[gnidx])
                    if temp_tot_partials>gen_num:
                        k_to_add = gen_num - copy_gen_num_partials[gnidx]
                        partial_inf_to_add = set(random.sample(partial_inf_answers[gnidx], k_to_add))

                    else:
                        partial_inf_to_add = partial_inf_answers[gnidx]
                    copy_gen_num_partials[gnidx]+=len(partial_inf_to_add)
                    answerstoadd = answerstoadd | partial_inf_to_add


            if max(len(answer_set - train_answer_set), len(train_answer_set - answer_set)) > max_ans_num:
                num_more_answer += 1
                continue
            #tot_to_gen = gen_num*(len(gen_num_partials)+1)
            rel_per_query_temp, anch_per_query_temp, new_set_rel = add_to_freq_dict(len(answerstoadd), set(rels),
                                                                       set(anchs), rel_per_query_overall[
                                                                           list2tuple(query_structure)].copy(),
                                                                       anch_per_query_overall[
                                                                           list2tuple(query_structure)].copy())
            if tot_qa_pairs > (tot_to_gen/ 10):
                top_k_rel_keys, top_k_rel_values = top_k_dict_values_sorting(rel_per_query_temp, 1)
                perc_top_rel = (top_k_rel_values[0] * 100) / (tot_qa_pairs + len(answerstoadd))

                top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_temp, 1)
                perc_top_anch = (top_k_anch_values[0] * 100) / (tot_qa_pairs + len(answerstoadd))
                if (perc_top_rel >= 20 and top_k_rel_keys[0] in new_set_rel) or (
                        perc_top_anch >= 20 and top_k_anch_keys[0] in anchs):
                    num_more_answer += 1
                    continue
            rel_per_query_overall[list2tuple(query_structure)] = rel_per_query_temp
            anch_per_query_overall[list2tuple(query_structure)] = anch_per_query_temp

            if len(answerstoadd)>0:
                queries[list2tuple(query_structure)].add(list2tuple(query))
                filtered_answers[list2tuple(query)] = answer_set - answerstoadd  # filtered answer set
                hard_answers[list2tuple(query)] = answerstoadd  # hard answer set

            num_sampled += n_sampled_step #cycle based on number of full-inference
            tot_qa_pairs+= len(answerstoadd)
            gen_num_partials = copy_gen_num_partials
            train_ans_num.append(len(filtered_answers[list2tuple(query)]))
            hard_ans_num.append(len(hard_answers[list2tuple(query)]))
        key_rel, perc_rel = get_top_k_frequency(dataset, tot_qa_pairs,rel_per_query_overall[list2tuple(query_structure)], 'rel')
        key_ent, perc_anch = get_top_k_frequency(dataset, tot_qa_pairs, anch_per_query_overall[list2tuple(query_structure)], 'ent')
        logging.info("Tot full-inf: " + str(num_sampled))
        logging.info("Tot qa-pairs: " + str(tot_qa_pairs))
        logging.info("Most present relation name: " + str(key_rel))
        logging.info("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_rel))
        logging.info("Most present anchor: " + str(key_ent))
        logging.info("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_anch))

    name_to_save = '%s-' % (mode)
    with open('./data/%s/%s-%s-propQA-queries.pkl' % (dataset, name_to_save,seed), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-%s-propQA-easy-answers.pkl' % (dataset, name_to_save,seed), 'wb') as f:
        pickle.dump(filtered_answers, f)
    with open('./data/%s/%s-%s-propQA-hard-answers.pkl' % (dataset, name_to_save,seed), 'wb') as f:
        pickle.dump(hard_answers, f)

    return queries, filtered_answers, hard_answers

def ground_queries_train(dataset, query_structures, ent_in, ent_out, small_ent_in, small_ent_out, gen_num, max_ans_num, query_names, mode, ent2id, rel2id):
    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
    tp_ans_num, fp_ans_num, fn_ans_num = [], [], []
    queries = defaultdict(set)
    tp_answers = defaultdict(set)
    fp_answers = defaultdict(set)
    fn_answers = defaultdict(set)
    s0 = time.time()
    old_num_sampled = -1
    for idx, query_structure in enumerate(query_structures):
        query_name = query_names[idx]
        num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
        while num_sampled < gen_num:
            if num_sampled != 0:
                if num_sampled % (gen_num//100) == 0 and num_sampled != old_num_sampled:
                    logging.info('%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s'%(mode,
                        query_structure,
                        num_sampled, gen_num, (time.time()-s0)/num_sampled, num_try, num_repeat, num_more_answer,
                        num_broken, num_no_extra_answer, num_no_extra_negative, num_empty))
                    old_num_sampled = num_sampled
            print ('%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s'%(mode,
                query_structure,
                num_sampled, gen_num, (time.time()-s0)/(num_sampled+0.001), num_try, num_repeat, num_more_answer,
                num_broken, num_no_extra_answer, num_no_extra_negative, num_empty), end='\r')
            num_try += 1
            empty_query_structure = deepcopy(query_structure)
            answer = random.sample(ent_in.keys(), 1)[0]
            broken_flag = fill_query(empty_query_structure, ent_in, ent_out, answer, ent2id, rel2id)
            if broken_flag:
                num_broken += 1
                continue
            query = empty_query_structure
            answer_set = achieve_answer(query, ent_in, ent_out)
            small_answer_set = achieve_answer(query, small_ent_in, small_ent_out)
            if len(answer_set) == 0:
                num_empty += 1
                continue
            if mode != 'train':
                if len(answer_set - small_answer_set) == 0:
                    num_no_extra_answer += 1
                    continue
                if 'n' in query_name:
                    if len(small_answer_set - answer_set) == 0:
                        num_no_extra_negative += 1
                        continue
            if max(len(answer_set - small_answer_set), len(small_answer_set - answer_set)) > max_ans_num:
                num_more_answer += 1
                continue
            if list2tuple(query) in queries[list2tuple(query_structure)]:
                num_repeat += 1
                continue
            queries[list2tuple(query_structure)].add(list2tuple(query))
            tp_answers[list2tuple(query)] = small_answer_set
            fp_answers[list2tuple(query)] = small_answer_set - answer_set
            fn_answers[list2tuple(query)] = answer_set - small_answer_set
            num_sampled += 1
    print ()
    name_to_save = '%s-'%(mode)
    with open('./data/%s/%s-queries.pkl'%(dataset, name_to_save), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-answers.pkl'%(dataset, name_to_save), 'wb') as f:
        pickle.dump(fn_answers, f)
    return queries, fn_answers

def generate_queries(dataset, query_structures, gen_num, max_ans_num, gen_train, gen_valid, gen_test, gen_num_per_query,query_names,
                     mode,seed):
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

    if query_structures[0] == ['e', ['r']]:
        if gen_train:
            write_links(dataset, train_ent_out, defaultdict(lambda: defaultdict(set)), max_ans_num, 'train-')
        if gen_valid:
            write_links(dataset, valid_only_ent_out, train_ent_out, max_ans_num, 'valid-')
        if gen_test:
            write_links(dataset, test_only_ent_out, valid_ent_out, max_ans_num, 'test-')
        print("link prediction created!")
        exit(-1)

    formatted_date_time = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    name_to_save = formatted_date_time + "_new_bench_" + str(dataset) + "_" + str(mode)
    set_logger("./data/{}/".format(dataset), name_to_save)
    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_empty = 0, 0, 0, 0, 0, 0
    train_ans_num = []
    s0 = time.time()
    if gen_train:
        train_queries, train_answers = ground_queries_train(dataset, query_structures,
                                                            train_ent_in, train_ent_out,
                                                            defaultdict(lambda: defaultdict(set)),
                                                            defaultdict(lambda: defaultdict(set)),
                                                            gen_num[0], gen_num_per_query,max_ans_num, query_names, 'train', ent2id,
                                                            rel2id)
    if gen_valid:
        valid_queries, valid_train_answers, valid_hard_answers = ground_queries(dataset, query_structures,
                                                                                valid_ent_in, valid_ent_out,
                                                                                train_ent_in, train_ent_out,
                                                                                valid_only_ent_in, valid_only_ent_out,
                                                                                gen_num[1], gen_num_per_query,max_ans_num, query_names,
                                                                                'valid', ent2id, rel2id,seed)
    if gen_test:
        test_queries, test_train_answers, test_hard_answers = ground_queries(dataset, query_structures,
                                                                             test_ent_in, test_ent_out, valid_ent_in,
                                                                             valid_ent_out, test_only_ent_in,
                                                                             test_only_ent_out, gen_num[2], gen_num_per_query,max_ans_num,
                                                                             query_names, 'test', ent2id, rel2id,seed)
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
@click.option('--mode', default=None)
@click.option('--index_only', is_flag=True, default=False)
@click.option('--fourp', is_flag=True, default=False)
def main(dataset, seed, gen_train_num, gen_valid_num, gen_test_num, max_ans_num, reindex, gen_train, gen_valid,
         gen_test, gen_id, mode, index_only, fourp):
    gen_num_per_query = { #number of reductions per query type. i.e. 2p has only one element, while 3p has 2 possible reductions
        "1p": [],
        "2p" : [0],
        "3p" : [0, 0],
        "4p": [0, 0, 0],
        "2i": [0],
        "3i": [0, 0],
        "4i": [0, 0, 0],
        "pi": [0, 0, 0],
        "ip": [0, 0, 0],
        "2u": [],
        "up": [0, 0],
        "2in": [],
        "3in": [0],
        "pin": [0],
        "pni": [],
        "inp": [0]
    }
    e = 'e'
    r = 'r'
    n = 'n'
    u = 'u'
    gen_train_num = 149010 #depends on the 1p queries of the specific benchmark
    if gen_train:
        query_structures = [
            #[e, [r]],
            [e, [r, r]],
            [e, [r, r, r]],
            [[e, [r]], [e, [r]]],
            [[e, [r]], [e, [r]], [e, [r]]],
            #[[e, [r, r]], [e, [r]]],
            #[[[e, [r]], [e, [r]]], [r]],
            # negation
            [[e, [r]], [e, [r, n]]],
            [[e, [r]], [e, [r]], [e, [r, n]]],
            [[e, [r, r]], [e, [r, n]]],
            [[e, [r, r, n]], [e, [r]]],
            [[[e, [r]], [e, [r, n]]], [r]],
            # union
            #[[e, [r]], [e, [r]], [u]],
            #[[[e, [r]], [e, [r]], [u]], [r]],
            # harder?
            # [e, [r, r, r, r]],
            # [[e, [r]], [e, [r]], [e, [r]], [e, [r]]],
        ]
        #query_names = ['2p', '3p', '2i', '3i', '2in', '3in', 'pin', 'pni', 'inp']
        query_names = ['3p', '2i', '3i', '2in', '3in', 'pin', 'pni', 'inp']
    else:
        query_structures = [
            #[e, [r]], #to be constructed separately
            [e, [r, r]],
            [e, [r, r, r]],
            [[e, [r]], [e, [r]]],
            [[e, [r]], [e, [r]], [e, [r]]],
            [[e, [r, r]], [e, [r]]],
            [[[e, [r]], [e, [r]]], [r]],
            # negation
            # [[e, [r]], [e, [r, n]]],
            [[e, [r]], [e, [r]], [e, [r, n]]], #3in
            [[e, [r, r]], [e, [r, n]]], #pin
            # [[e, [r, r, n]], [e, [r]]],
            [[[e, [r]], [e, [r, n]]], [r]], #inp
            # union
            [[e, [r]], [e, [r]], [u]],
            [[[e, [r]], [e, [r]], [u]], [r]],
            # harder?
            [e, [r, r, r, r]],
            [[e, [r]], [e, [r]], [e, [r]], [e, [r]]],
        ]
        query_names = ['2p','3p', '2i', '3i', 'pi', 'ip', '3in', 'pin', 'inp', '2u', 'up', '4p', '4i']

        
        gen_test_num = gen_valid_num = 10000
        if gen_test:
            mode = 'test'
        elif gen_valid:
            mode='valid'
    generate_queries(dataset, query_structures, [gen_train_num, gen_valid_num, gen_test_num], max_ans_num, gen_train,
                     gen_valid, gen_test, gen_num_per_query,query_names, mode,seed)


if __name__ == '__main__':
    main()