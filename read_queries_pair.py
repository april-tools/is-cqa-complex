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
    # files = ['train.txt']
    # indexified_files = ['train_indexified.txt']
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

                if e1 in ent2id.keys() and e2 in ent2id.keys():
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


def write_links(dataset, ent_out, small_ent_out, max_ans_num, name):
    queries = defaultdict(set)
    tp_answers = defaultdict(set)
    fn_answers = defaultdict(set)
    fp_answers = defaultdict(set)
    num_more_answer = 0
    for ent in ent_out:
        for rel in ent_out[ent]:
            if len(ent_out[ent][rel]) <= max_ans_num:
                queries[('e', ('r',))].add((ent, (rel,)))
                tp_answers[(ent, (rel,))] = small_ent_out[ent][rel]
                fn_answers[(ent, (rel,))] = ent_out[ent][rel]
            else:
                num_more_answer += 1

    with open('./data/%s/%s-queries.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-tp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(tp_answers, f)
    with open('./data/%s/%s-fn-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(fn_answers, f)
    with open('./data/%s/%s-fp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(fp_answers, f)
    print(num_more_answer)


def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl_file(directory_path,filenames,datas):
    os.makedirs(directory_path)
    for filename,data in zip(filenames,datas):
        filename = directory_path + "/" + filename
        with open(filename + "queries.pkl", 'wb') as f:
            pickle.dump(data[0], f)
        with open(filename + "filters.pkl", 'wb') as f:
            pickle.dump(data[1], f)
        with open(filename + "answers.pkl", 'wb') as f:
            pickle.dump(data[2], f)

def find_answers(query_structure, queries, missing_ent_in, missing_ent_out, all_ent_in, all_ent_out, easy_ent_in,
                 easy_ent_out, mode,dataset):
    '''
    missing_ent = entities related only to the validation/test set
    all_ent = entities related to the train + validation/test set
    easy_ent = entities related only to train

    '''
    random.seed(0)
    num_sampled = 0
    folder_name = mode + "-query-reduction"
    filepath = "./data/{}/{}".format(dataset, folder_name)
    queries_2p_2p, queries_2p_1p = defaultdict(set), defaultdict(set)
    answers_2p_2p, answers_2p_2p_filters, answers_2p_1p, answers_2p_1p_filters = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_1p_1p = defaultdict(set)
    answers_1p_1p, answers_1p_1p_filters = defaultdict(set), defaultdict(set)

    queries_3p_3p ,queries_3p_2p , queries_3p_1p = defaultdict(set), defaultdict(set),defaultdict(set)
    answers_3p_3p , answers_3p_3p_filters , answers_3p_2p , answers_3p_2p_filters ,answers_3p_1p , answers_3p_1p_filters= defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_2i_2i , queries_2i_1p = defaultdict(set),defaultdict(set)
    answers_2i_2i , answers_2i_2i_filters , answers_2i_1p , answers_2i_1p_filters = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_3i_3i ,queries_3i_2i , queries_3i_1p = defaultdict(set),defaultdict(set),defaultdict(set)
    answers_3i_3i , answers_3i_3i_filters , answers_3i_2i , answers_3i_2i_filters , answers_3i_1p , answers_3i_1p_filters= defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_pi_pi , queries_pi_2i , queries_pi_2p , queries_pi_1p= defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)
    answers_pi_pi , answers_pi_pi_filters , answers_pi_2i , answers_pi_2i_filters , answers_pi_2p , answers_pi_2p_filters , answers_pi_1p , answers_pi_1p_filters = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_ip_ip , queries_ip_2i , queries_ip_2p , queries_ip_1p = defaultdict(set), defaultdict(set),defaultdict(set),defaultdict(set)
    answers_ip_ip , answers_ip_ip_filters , answers_ip_2i , answers_ip_2i_filters , answers_ip_2p , answers_ip_2p_filters , answers_ip_1p , answers_ip_1p_filters = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_up_up , queries_up_2u , queries_up_2p , queries_up_1p = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)
    answers_up_up , answers_up_up_filters , answers_up_2u , answers_up_2u_filters , answers_up_2p , answers_up_2p_filters , answers_up_1p , answers_up_1p_filters = defaultdict(
        set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    n_1p_1p = 0
    n_1p_2p = n_2p_2p = 0
    n_1p_3p = n_2p_3p = n_3p_3p = 0
    n_2i_1i = n_2i_2i = n_3i_1i = n_3i_2i = n_3i_3i = 0
    n_pi_1p = n_pi_2p = n_pi_2i = n_pi_pi = 0
    n_ip_1p = n_ip_2p = n_ip_2i = n_ip_ip = 0
    n_up_1p = n_up_2p = n_up_2u = n_up_up = 0
    n_tot_hard_answers_1p = n_tot_hard_answers_2p = n_tot_hard_answers_3p = n_tot_hard_answers_2i = n_tot_hard_answers_3i = n_tot_hard_answers_pi = n_tot_hard_answers_ip = n_tot_hard_answers_up = 0
    for query in queries:
        query = tuple2list(query)
        answer_set = achieve_answer(query, all_ent_in, all_ent_out)
        easy_answer_set = achieve_answer(query, easy_ent_in, easy_ent_out)
        hard_answer_set = answer_set - easy_answer_set  # we take into account only the non-trivial answers

        if query_structure == ['e', ['r']]:
            n_tot_hard_answers_1p+=len(hard_answer_set)
            entity = query[0]
            rel = query[1][0]
            reachable_answers_1p = missing_ent_out[entity][rel]
            if len(reachable_answers_1p) > 0:
                queries_1p_1p[list2tuple(query_structure)].add(list2tuple(query))
                answers_1p_1p_filters[list2tuple(query)] = answer_set - reachable_answers_1p
                answers_1p_1p[list2tuple(query)] = reachable_answers_1p
            n_1p_1p+=len(reachable_answers_1p)
        if query_structure == ['e', ['r', 'r']]:
            n_tot_hard_answers_2p += len(hard_answer_set)
            # check if answer reachable with training link
            entity = query[0]
            rel1 = query[1][0]
            rel2 = query[1][1]

            # compute the intermediate answers of the query (entity,rel1,?x) on the training graph
            answer_set_q2_1 = set()
            answer_set_q1_1 = easy_ent_out[entity][rel1]
            # for each intermediate answer, check if there is a test link that brings you to an answer (?x, rel2, ?y)
            for ent_int in answer_set_q1_1:
                answer_set_q2_1.update(missing_ent_out[ent_int][rel2])

            # opposite as before; test for intermediate an train for final step
            answer_set_q2_2 = set()
            answer_set_q1_2 = missing_ent_out[entity][rel1]
            for ent_int in answer_set_q1_2:
                answer_set_q2_2.update(easy_ent_out[ent_int][rel2])

            reachable_answers1p = (answer_set_q2_1 | answer_set_q2_2) - easy_answer_set
            n_1p_2p += len(reachable_answers1p)
            reachable_answers2p = set()
            if len(reachable_answers1p) < len(hard_answer_set):
                answer_set_q2_3 = set()
                answer_set_q1_3 = missing_ent_out[entity][rel1]
                for ent_int in answer_set_q1_3:
                    answer_set_q2_3.update(missing_ent_out[ent_int][rel2])
                reachable_answers2p = answer_set_q2_3 - reachable_answers1p - easy_answer_set
                n_2p_2p += len(reachable_answers2p)
                queries_2p_2p[list2tuple(query_structure)].add(list2tuple(query))
                answers_2p_2p_filters[list2tuple(query)] = reachable_answers1p | easy_answer_set
                answers_2p_2p[list2tuple(query)] = reachable_answers2p
            if len(reachable_answers1p) > 0:
                queries_2p_1p[list2tuple(query_structure)].add(list2tuple(query))
                answers_2p_1p_filters[list2tuple(query)] = easy_answer_set | reachable_answers2p
                answers_2p_1p[list2tuple(query)] = reachable_answers1p

        if query_structure == ['e', ['r', 'r', 'r']]:
            n_tot_hard_answers_3p += len(hard_answer_set)
            # 0 existing 1 predicted
            # existing + existing + predicted --> 1p 001
            # existing + predicted + existing --> 1p 010
            # existing + predicted + predicted --> 2p 011
            # predicted + existing + existing --> 1p 100
            # predicted + predicted + existing --> 2p 110
            # predicted + existing + predicted --> 2p 101
            # predicted + predicted + predicted --> 3p 111
            entity = query[0]
            rel1 = query[1][0]
            rel2 = query[1][1]
            rel3 = query[1][2]

            # existing + existing + predicted --> 1p 001
            answer_set_q2_1 = set()
            answer_set_q3_1 = set()
            answer_set_q1_1 = easy_ent_out[entity][rel1]
            # for each intermediate answer, check if there is a test link that brings you to an answer (?x, rel2, ?y)
            for ent_int in answer_set_q1_1:
                answer_set_q2_1.update(easy_ent_out[ent_int][rel2])
            for ent_int_int in answer_set_q2_1:
                answer_set_q3_1.update(missing_ent_out[ent_int_int][rel3])

            # existing + predicted + existing --> 1p 010
            answer_set_q2_2 = set()
            answer_set_q3_2 = set()
            answer_set_q1_2 = easy_ent_out[entity][rel1]
            # for each intermediate answer, check if there is a test link that brings you to an answer (?x, rel2, ?y)
            for ent_int in answer_set_q1_2:
                answer_set_q2_2.update(missing_ent_out[ent_int][rel2])
            for ent_int_int in answer_set_q2_2:
                answer_set_q3_2.update(easy_ent_out[ent_int_int][rel3])

            # predicted + existing + existing --> 1p 100
            answer_set_q2_3 = set()
            answer_set_q3_3 = set()
            answer_set_q1_3 = missing_ent_out[entity][rel1]
            # for each intermediate answer, check if there is a test link that brings you to an answer (?x, rel2, ?y)
            for ent_int in answer_set_q1_3:
                answer_set_q2_3.update(easy_ent_out[ent_int][rel2])
            for ent_int_int in answer_set_q2_3:
                answer_set_q3_3.update(easy_ent_out[ent_int_int][rel3])

            reachable_answers_1p = (answer_set_q3_1 | answer_set_q3_2 | answer_set_q3_3) - easy_answer_set  # subtract the easy answers
            n_1p_3p += len(reachable_answers_1p)
            if len(reachable_answers_1p)>0:
                queries_3p_1p[list2tuple(query_structure)].add(list2tuple(query))
                answers_3p_1p_filters[list2tuple(query)] = answer_set - reachable_answers_1p
                answers_3p_1p[list2tuple(query)] = reachable_answers_1p
            if len(reachable_answers_1p) < len(hard_answer_set):
                # continue the computation for 2p/3p
                # existing + predicted + predicted --> 2p 011
                answer_set_q2_4 = set()
                answer_set_q3_4 = set()
                answer_set_q1_4 = easy_ent_out[entity][rel1]
                for ent_int in answer_set_q1_4:
                    answer_set_q2_4.update(missing_ent_out[ent_int][rel2])
                for ent_int_int in answer_set_q2_4:
                    answer_set_q3_4.update(missing_ent_out[ent_int_int][rel3])
                # predicted + predicted + existing --> 2p 110
                answer_set_q2_5 = set()
                answer_set_q3_5 = set()
                answer_set_q1_5 = missing_ent_out[entity][rel1]
                for ent_int in answer_set_q1_5:
                    answer_set_q2_5.update(missing_ent_out[ent_int][rel2])
                for ent_int_int in answer_set_q2_5:
                    answer_set_q3_5.update(easy_ent_out[ent_int_int][rel3])
                # predicted + existing + predicted --> 2p 101
                answer_set_q2_7 = set()
                answer_set_q3_7 = set()
                answer_set_q1_7 = missing_ent_out[entity][rel1]
                for ent_int in answer_set_q1_7:
                    answer_set_q2_7.update(easy_ent_out[ent_int][rel2])
                for ent_int_int in answer_set_q2_7:
                    answer_set_q3_7.update(missing_ent_out[ent_int_int][rel3])
                reachable_answers_2p = (answer_set_q3_4 | answer_set_q3_5 | answer_set_q3_7)  - reachable_answers_1p - easy_answer_set  # subtract the easy answers and the 1p answers
                n_2p_3p += len(reachable_answers_2p)
                if len(reachable_answers_2p)>0:
                    queries_3p_2p[list2tuple(query_structure)].add(list2tuple(query))
                    answers_3p_2p_filters[list2tuple(query)] = answer_set - reachable_answers_2p
                    answers_3p_2p[list2tuple(query)] = reachable_answers_2p
                if len(reachable_answers_1p | reachable_answers_2p) < len(hard_answer_set):
                    # predicted + predicted + existing --> 3p 111
                    answer_set_q2_6 = set()
                    answer_set_q3_6 = set()
                    answer_set_q1_6 = missing_ent_out[entity][rel1]
                    for ent_int in answer_set_q1_6:
                        answer_set_q2_6.update(missing_ent_out[ent_int][rel2])
                    for ent_int_int in answer_set_q2_6:
                        answer_set_q3_6.update(missing_ent_out[ent_int_int][rel3])
                    reachable_answers_3p = answer_set_q3_6 - reachable_answers_2p - reachable_answers_1p - easy_answer_set  # subtract the easy answers and the 1p/2p answers
                    n_3p_3p += len(reachable_answers_3p)
                    if len(reachable_answers_3p) >0:
                        queries_3p_3p[list2tuple(query_structure)].add(list2tuple(query))
                        answers_3p_3p_filters[list2tuple(query)] = answer_set - reachable_answers_3p
                        answers_3p_3p[list2tuple(query)] = reachable_answers_3p
        if query_structure == [['e', ['r']], ['e', ['r']]]:
            n_tot_hard_answers_2i += len(hard_answer_set)
            # 2i
            entity1 = query[0][0]
            rel1 = query[0][1][0]
            entity2 = query[1][0]
            rel2 = query[1][1][0]
            # 01
            # compute the answers of the query (entity1,rel1,?y) on the training graph
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            # compute the answers of the query (entity2,rel2,?y) on the missing graph
            answer_set_q1_2 = missing_ent_out[entity2][rel2]
            answers_01 = answer_set_q1_1 & answer_set_q1_2
            # 10
            # compute the answers of the query (entity1,rel1,?y) on the missing graph
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            # compute the answers of the query (entity2,rel2,?y) on the training graph
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answers_10 = answer_set_q1_1 & answer_set_q1_2

            reachable_answers_1p = (answers_01 | answers_10) - easy_answer_set
            n_2i_1i += len(reachable_answers_1p)
            if len(reachable_answers_1p)>0:
                queries_2i_1p[list2tuple(query_structure)].add(list2tuple(query))
                answers_2i_1p_filters[list2tuple(query)] = answer_set - reachable_answers_1p
                answers_2i_1p[list2tuple(query)] = reachable_answers_1p
            if len(reachable_answers_1p) < len(hard_answer_set):
                # 11
                # compute the answers of the query (entity1,rel1,?y) on the missing graph
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                # compute the answers of the query (entity2,rel2,?y) on the missing graph
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answers_11 = answer_set_q1_1 & answer_set_q1_2
                reachable_answers_2i = answers_11 - reachable_answers_1p - easy_answer_set
                n_2i_2i += len(reachable_answers_2i)
                if len(reachable_answers_2i) > 0:
                    queries_2i_2i[list2tuple(query_structure)].add(list2tuple(query))
                    answers_2i_2i_filters[list2tuple(query)] = answer_set - reachable_answers_2i
                    answers_2i_2i[list2tuple(query)] = reachable_answers_2i
        if query_structure == [['e', ['r']], ['e', ['r']], ['e', ['r']]]:
            n_tot_hard_answers_3i += len(hard_answer_set)
            # 3i
            # 001
            # 010
            # 011
            # 100
            # 101
            # 110
            # 111
            entity1 = query[0][0]
            rel1 = query[0][1][0]
            entity2 = query[1][0]
            rel2 = query[1][1][0]
            entity3 = query[2][0]
            rel3 = query[2][1][0]
            # 001
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answer_set_q1_3 = missing_ent_out[entity3][rel3]
            answers_001 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3
            # 010
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = missing_ent_out[entity2][rel2]
            answer_set_q1_3 = easy_ent_out[entity3][rel3]
            answers_010 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3
            # 100
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answer_set_q1_3 = easy_ent_out[entity3][rel3]
            answers_100 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3
            reachable_answers_1p = (answers_001 | answers_010 | answers_100) - easy_answer_set
            n_3i_1i += len(reachable_answers_1p)
            if len(reachable_answers_1p)>0:
                queries_3i_1p[list2tuple(query_structure)].add(list2tuple(query))
                answers_3i_1p_filters[list2tuple(query)] = answer_set - reachable_answers_1p
                answers_3i_1p[list2tuple(query)] = reachable_answers_1p
            if len(reachable_answers_1p) < len(hard_answer_set):
                # 011
                answer_set_q1_1 = easy_ent_out[entity1][rel1]
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answer_set_q1_3 = missing_ent_out[entity3][rel3]
                answers_011 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3

                # 101
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                answer_set_q1_2 = easy_ent_out[entity2][rel2]
                answer_set_q1_3 = missing_ent_out[entity3][rel3]
                answers_101 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3

                # 110
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answer_set_q1_3 = easy_ent_out[entity3][rel3]
                answers_110 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3

                reachable_answers_2i = (
                                               answers_011 | answers_101 | answers_110) - reachable_answers_1p - easy_answer_set
                n_3i_2i += len(reachable_answers_2i)
                if len(reachable_answers_2i) > 0:
                    queries_3i_2i[list2tuple(query_structure)].add(list2tuple(query))
                    answers_3i_2i_filters[list2tuple(query)] = answer_set - reachable_answers_2i
                    answers_3i_2i[list2tuple(query)] = reachable_answers_2i
                if len(reachable_answers_2i | reachable_answers_1p) < len(hard_answer_set):
                    # 111
                    answer_set_q1_1 = missing_ent_out[entity1][rel1]
                    answer_set_q1_2 = missing_ent_out[entity2][rel2]
                    answer_set_q1_3 = missing_ent_out[entity3][rel3]
                    answers_111 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3

                    reachable_answers_3i = (answers_111 - reachable_answers_1p - reachable_answers_2i) - easy_answer_set
                    n_3i_3i += len(reachable_answers_3i)
                    if len(reachable_answers_3i) > 0:
                        queries_3i_3i[list2tuple(query_structure)].add(list2tuple(query))
                        answers_3i_3i_filters[list2tuple(query)] = answer_set - reachable_answers_3i
                        answers_3i_3i[list2tuple(query)] = reachable_answers_3i
        if query_structure == [['e', ['r', 'r']], ['e', ['r']]]:
            # ((e1,r1.?x) and (?x,r2,?y))and(e2,r3,?y)
            n_tot_hard_answers_pi += len(hard_answer_set)
            entity1 = query[0][0]
            rel1 = query[0][1][0]
            rel2 = query[0][1][1]
            entity2 = query[1][0]
            rel3 = query[1][1][0]
            # 001
            answer_set_q1_2 = set()
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            for ent_int in answer_set_q1_1:
                answer_set_q1_2.update(easy_ent_out[ent_int][rel2])
            answer_set_q1_3 = missing_ent_out[entity2][rel3]
            answers_001 = answer_set_q1_2 & answer_set_q1_3
            # 010
            answer_set_q1_2 = set()
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            for ent_int in answer_set_q1_1:
                answer_set_q1_2.update(missing_ent_out[ent_int][rel2])
            answer_set_q1_3 = easy_ent_out[entity2][rel3]
            answers_010 = answer_set_q1_2 & answer_set_q1_3
            # 100
            answer_set_q1_2 = set()
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            for ent_int in answer_set_q1_1:
                answer_set_q1_2.update(easy_ent_out[ent_int][rel2])
            answer_set_q1_3 = easy_ent_out[entity2][rel3]
            answers_100 = answer_set_q1_2 & answer_set_q1_3
            reachable_answers_1p = (answers_001 | answers_010 | answers_100) - easy_answer_set
            n_pi_1p += len(reachable_answers_1p)
            if len(reachable_answers_1p) > 0:
                queries_pi_1p[list2tuple(query_structure)].add(list2tuple(query))
                answers_pi_1p_filters[list2tuple(query)] = answer_set - reachable_answers_1p
                answers_pi_1p[list2tuple(query)] = reachable_answers_1p
            if len(reachable_answers_1p) < len(hard_answer_set):
                # 2i
                # 011
                answer_set_q1_2 = set()
                answer_set_q1_1 = easy_ent_out[entity1][rel1]
                for ent_int in answer_set_q1_1:
                    answer_set_q1_2.update(missing_ent_out[ent_int][rel2])
                answer_set_q1_3 = missing_ent_out[entity2][rel3]
                answers_011 = answer_set_q1_2 & answer_set_q1_3
                # 101
                answer_set_q1_2 = set()
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                for ent_int in answer_set_q1_1:
                    answer_set_q1_2.update(easy_ent_out[ent_int][rel2])
                answer_set_q1_3 = missing_ent_out[entity2][rel3]
                answers_101 = answer_set_q1_2 & answer_set_q1_3
                reachable_answers_2i = (answers_011 | answers_101) - reachable_answers_1p - easy_answer_set
                n_pi_2i += len(reachable_answers_2i)
                if len(reachable_answers_2i) > 0:
                    queries_pi_2i[list2tuple(query_structure)].add(list2tuple(query))
                    answers_pi_2i_filters[list2tuple(query)] = answer_set - reachable_answers_2i
                    answers_pi_2i[list2tuple(query)] = reachable_answers_2i
                # 2p
                # 110
                answer_set_q1_2 = set()
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                for ent_int in answer_set_q1_1:
                    answer_set_q1_2.update(missing_ent_out[ent_int][rel2])
                answer_set_q1_3 = easy_ent_out[entity2][rel3]
                answers_110 = answer_set_q1_2 & answer_set_q1_3
                reachable_answers_2p = answers_110 - reachable_answers_1p - reachable_answers_2i - easy_answer_set
                n_pi_2p += len(reachable_answers_2p)
                if len(reachable_answers_2p) > 0:
                    queries_pi_2p[list2tuple(query_structure)].add(list2tuple(query))
                    answers_pi_2p_filters[list2tuple(query)] = answer_set - reachable_answers_2p
                    answers_pi_2p[list2tuple(query)] = reachable_answers_2p
                if len(reachable_answers_2p | reachable_answers_2i | reachable_answers_1p) < len(hard_answer_set):
                    # 111
                    # pi
                    answer_set_q1_2 = set()
                    answer_set_q1_1 = missing_ent_out[entity1][rel1]
                    for ent_int in answer_set_q1_1:
                        answer_set_q1_2.update(missing_ent_out[ent_int][rel2])
                    answer_set_q1_3 = missing_ent_out[entity2][rel3]
                    answers_111 = answer_set_q1_2 & answer_set_q1_3
                    reachable_answers_pi = answers_111 - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i - easy_answer_set
                    n_pi_pi += len(reachable_answers_pi)
                    if len(reachable_answers_pi) > 0:
                        queries_pi_pi[list2tuple(query_structure)].add(list2tuple(query))
                        answers_pi_pi_filters[list2tuple(query)] = answer_set - reachable_answers_pi
                        answers_pi_pi[list2tuple(query)] = reachable_answers_pi
        if query_structure == [[['e', ['r']], ['e', ['r']]], ['r']]:
            # (e1,r1,?x)and(e2,r2,?x)and(?x,r3,?y)
            n_tot_hard_answers_ip += len(hard_answer_set)
            entity1 = query[0][0][0]
            rel1 = query[0][0][1][0]
            entity2 = query[0][1][0]
            rel2 = query[0][1][1][0]
            rel3 = query[1][0]
            # 001
            answers_001 = set()
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answers_00 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_00:
                answers_001.update(missing_ent_out[ele][rel3])
            # 010
            answers_010 = set()
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = missing_ent_out[entity2][rel2]
            answers_01 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_01:
                answers_010.update(easy_ent_out[ele][rel3])
            # 100
            answers_100 = set()
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answers_10 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_10:
                answers_100.update(easy_ent_out[ele][rel3])
            reachable_answers_1p = (answers_001 | answers_010 | answers_100) - easy_answer_set
            n_ip_1p += len(reachable_answers_1p)
            if len(reachable_answers_1p) > 0:
                queries_ip_1p[list2tuple(query_structure)].add(list2tuple(query))
                answers_ip_1p_filters[list2tuple(query)] = answer_set - reachable_answers_1p
                answers_ip_1p[list2tuple(query)] = reachable_answers_1p
            if len(reachable_answers_1p) < len(hard_answer_set):
                # compute 2p and 2i
                # 2i
                # 110
                answers_110 = set()
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answers_11 = answer_set_q1_1 & answer_set_q1_2
                for ele in answers_11:
                    answers_110.update(easy_ent_out[ele][rel3])
                reachable_answers_2i = answers_110 - reachable_answers_1p - easy_answer_set
                n_ip_2i += len(reachable_answers_2i)
                if len(reachable_answers_2i) > 0:
                    queries_ip_2i[list2tuple(query_structure)].add(list2tuple(query))
                    answers_ip_2i_filters[list2tuple(query)] = answer_set - reachable_answers_2i
                    answers_ip_2i[list2tuple(query)] = reachable_answers_2i
                # 2p
                # 101
                answers_101 = set()
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                answer_set_q1_2 = easy_ent_out[entity2][rel2]
                answers_10 = answer_set_q1_1 & answer_set_q1_2
                for ele in answers_10:
                    answers_101.update(missing_ent_out[ele][rel3])

                # 011
                answers_011 = set()
                answer_set_q1_1 = easy_ent_out[entity1][rel1]
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answers_01 = answer_set_q1_1 & answer_set_q1_2
                for ele in answers_01:
                    answers_011.update(missing_ent_out[ele][rel3])
                reachable_answers_2p = (answers_101 | answers_011) - reachable_answers_1p - reachable_answers_2i - easy_answer_set
                n_ip_2p += len(reachable_answers_2p)
                if len(reachable_answers_2p) > 0:
                    queries_ip_2p[list2tuple(query_structure)].add(list2tuple(query))
                    answers_ip_2p_filters[list2tuple(query)] = answer_set - reachable_answers_2p
                    answers_ip_2p[list2tuple(query)] = reachable_answers_2p
                if len(reachable_answers_2p | reachable_answers_2i | reachable_answers_1p) < len(hard_answer_set):
                    # 111
                    answers_111 = set()
                    answer_set_q1_1 = missing_ent_out[entity1][rel1]
                    answer_set_q1_2 = missing_ent_out[entity2][rel2]
                    answers_11 = answer_set_q1_1 & answer_set_q1_2
                    for ele in answers_11:
                        answers_111.update(missing_ent_out[ele][rel3])
                    reachable_answers_ip = answers_111 - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i - easy_answer_set
                    n_ip_ip += len(reachable_answers_ip)
                    if len(reachable_answers_ip) > 0:
                        queries_ip_ip[list2tuple(query_structure)].add(list2tuple(query))
                        answers_ip_ip_filters[list2tuple(query)] = answer_set - reachable_answers_ip
                        answers_ip_ip[list2tuple(query)] = reachable_answers_ip
        if query_structure == [[['e', ['r']], ['e', ['r']], ['u']], ['r']]:
            # (e1,r1,?x)OR(e2,r2,?x)and(?x,r3,?y)
            n_tot_hard_answers_up += len(hard_answer_set)
            entity1 = query[0][0][0]
            rel1 = query[0][0][1][0]
            entity2 = query[0][1][0]
            rel2 = query[0][1][1][0]
            rel3 = query[1][0]
            # 001
            answers_001 = set()
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answers_00 = answer_set_q1_1 | answer_set_q1_2
            for ele in answers_00:
                answers_001.update(missing_ent_out[ele][rel3])
            # 010
            answers_010 = set()
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = missing_ent_out[entity2][rel2]
            answers_01 = answer_set_q1_1 | answer_set_q1_2
            for ele in answers_01:
                answers_010.update(easy_ent_out[ele][rel3])
            # 100
            answers_100 = set()
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answers_10 = answer_set_q1_1 | answer_set_q1_2
            for ele in answers_10:
                answers_100.update(easy_ent_out[ele][rel3])
            reachable_answers_1p = (answers_001 | answers_010 | answers_100) - easy_answer_set
            n_up_1p += len(reachable_answers_1p)
            if len(reachable_answers_1p) > 0:
                queries_up_1p[list2tuple(query_structure)].add(list2tuple(query))
                answers_up_1p_filters[list2tuple(query)] = answer_set - reachable_answers_1p
                answers_up_1p[list2tuple(query)] = reachable_answers_1p
            if len(reachable_answers_1p) < len(hard_answer_set):
                # compute 2p and 2u
                # 2u
                # 110
                answers_110 = set()
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answers_11 = answer_set_q1_1 | answer_set_q1_2
                for ele in answers_11:
                    answers_110.update(easy_ent_out[ele][rel3])
                reachable_answers_2u = answers_110 - reachable_answers_1p - easy_answer_set
                n_up_2u += len(reachable_answers_2u)

                if len(reachable_answers_2u) > 0:
                    queries_up_2u[list2tuple(query_structure)].add(list2tuple(query))
                    answers_up_2u_filters[list2tuple(query)] = answer_set - reachable_answers_2u
                    answers_up_2u[list2tuple(query)] = reachable_answers_2u

                # 2p
                # 101
                answers_101 = set()
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                answer_set_q1_2 = easy_ent_out[entity2][rel2]
                answers_10 = answer_set_q1_1 | answer_set_q1_2
                for ele in answers_10:
                    answers_101.update(missing_ent_out[ele][rel3])



                # 011
                answers_011 = set()
                answer_set_q1_1 = easy_ent_out[entity1][rel1]
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answers_01 = answer_set_q1_1 | answer_set_q1_2
                for ele in answers_01:
                    answers_011.update(missing_ent_out[ele][rel3])
                reachable_answers_2p = (answers_101 | answers_011) - reachable_answers_1p - reachable_answers_2u - easy_answer_set
                n_up_2p += len(reachable_answers_2p)
                if len(reachable_answers_2p) > 0:
                    queries_up_2p[list2tuple(query_structure)].add(list2tuple(query))
                    answers_up_2p_filters[list2tuple(query)] = answer_set - reachable_answers_2p
                    answers_up_2p[list2tuple(query)] = reachable_answers_2p

                if len(reachable_answers_2p | reachable_answers_2u | reachable_answers_1p) < len(hard_answer_set):
                    # 111
                    answers_111 = set()
                    answer_set_q1_1 = missing_ent_out[entity1][rel1]
                    answer_set_q1_2 = missing_ent_out[entity2][rel2]
                    answers_11 = answer_set_q1_1 | answer_set_q1_2
                    for ele in answers_11:
                        answers_111.update(missing_ent_out[ele][rel3])
                    reachable_answers_up = answers_111 - reachable_answers_1p - reachable_answers_2p - reachable_answers_2u - easy_answer_set
                    n_up_up += len(reachable_answers_up)
                    if len(reachable_answers_up) > 0:
                        queries_up_up[list2tuple(query_structure)].add(list2tuple(query))
                        answers_up_up_filters[list2tuple(query)] = answer_set - reachable_answers_up
                        answers_up_up[list2tuple(query)] = reachable_answers_up
        num_sampled += 1


    if n_1p_1p != 0:
        print("----1p----")
        print("Number of answers 1p: " + str(n_1p_1p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_1p))
        #writefiles
        directory_path = filepath + "/" + "1p"
        filenames = ["1p1p_"]
        datas = [[queries_1p_1p,answers_1p_1p_filters,answers_1p_1p]]
        save_pkl_file(directory_path,filenames,datas)
    if n_1p_2p != 0:
        print("----2p----")
        print("Number of answers 2p: " + str(n_2p_2p))
        print("Number of answers 1p: " + str(n_1p_2p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_2p))
        #writefiles
        directory_path = filepath + "/" + "2p"
        filenames = ["2p2p_","2p1p_"]
        datas = [[queries_2p_2p,answers_2p_2p_filters,answers_2p_2p],[queries_2p_1p,answers_2p_1p_filters,answers_2p_1p]]
        save_pkl_file(directory_path,filenames,datas)
    if n_1p_3p != 0:
        print("----3p----")
        print("Number of answers 3p: " + str(n_3p_3p))
        print("Number of answers 2p: " + str(n_2p_3p))
        print("Number of answers 1p: " + str(n_1p_3p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_3p))
        # writefiles
        directory_path = filepath + "/" + "3p"
        filenames = ["3p3p_", "3p2p_", "3p1p_"]
        datas = [[queries_3p_3p, answers_3p_3p_filters, answers_3p_3p],
                 [queries_3p_2p, answers_3p_2p_filters, answers_3p_2p],
                 [queries_3p_1p, answers_3p_1p_filters, answers_3p_1p]]
        save_pkl_file(directory_path, filenames, datas)
    if n_2i_1i != 0:
        print("----2i----")
        print("Number of answers 2i: " + str(n_2i_2i))
        print("Number of answers 1p: " + str(n_2i_1i))
        print("Total number of hard answers: " + str(n_tot_hard_answers_2i))
        # writefiles
        directory_path = filepath + "/" + "2i"
        filenames = ["2i2i_", "2i1p_"]
        datas = [[queries_2i_2i, answers_2i_2i_filters, answers_2i_2i],
                 [queries_2i_1p, answers_2i_1p_filters, answers_2i_1p]]
        save_pkl_file(directory_path, filenames, datas)
    if n_3i_1i != 0:
        print("----3i----")
        print("Number of answers 3i: " + str(n_3i_3i))
        print("Number of answers 2i: " + str(n_3i_2i))
        print("Number of answers 1p: " + str(n_3i_1i))
        print("Total number of hard answers: " + str(n_tot_hard_answers_3i))
        # writefiles
        directory_path = filepath + "/" + "3i"
        filenames = ["3i3i_", "3i2i_", "3i1p_"]
        datas = [[queries_3i_3i, answers_3i_3i_filters, answers_3i_3i],
                 [queries_3i_2i, answers_3i_2i_filters, answers_3i_2i],
                 [queries_3i_1p, answers_3i_1p_filters, answers_3i_1p]]
        save_pkl_file(directory_path, filenames, datas)
    if n_pi_1p != 0:
        print("----pi----")
        print("Number of answers pi: " + str(n_pi_pi))
        print("Number of answers 2i: " + str(n_pi_2i))
        print("Number of answers 2p: " + str(n_pi_2p))
        print("Number of answers 1p: " + str(n_pi_1p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_pi))
        directory_path = filepath + "/" + "pi"
        filenames = ["pipi_", "pi2p_", "pi2i_", "pi1p_"]
        datas = [[queries_pi_pi, answers_pi_pi_filters, answers_pi_pi],
                 [queries_pi_2p, answers_pi_2p_filters, answers_pi_2p],
                 [queries_pi_2i, answers_pi_2i_filters, answers_pi_2i],
                 [queries_pi_1p, answers_pi_1p_filters, answers_pi_1p]]
        save_pkl_file(directory_path, filenames, datas)
    if n_ip_1p != 0:
        print("----ip----")
        print("Number of answers ip: " + str(n_ip_ip))
        print("Number of answers 2i: " + str(n_ip_2i))
        print("Number of answers 2p: " + str(n_ip_2p))
        print("Number of answers 1p: " + str(n_ip_1p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_ip))
        directory_path = filepath + "/" + "ip"
        filenames = ["ipip_", "ip2p_", "ip2i_", "ip1p_"]
        datas = [[queries_ip_ip, answers_ip_ip_filters, answers_ip_ip],
                 [queries_ip_2p, answers_ip_2p_filters, answers_ip_2p],
                 [queries_ip_2i, answers_ip_2i_filters, answers_ip_2i],
                 [queries_ip_1p, answers_ip_1p_filters, answers_ip_1p]]
        save_pkl_file(directory_path, filenames, datas)
    if n_up_1p != 0:
        print("----up----")
        print("Number of answers up: " + str(n_up_up))
        print("Number of answers 2u: " + str(n_up_2u))
        print("Number of answers 2p: " + str(n_up_2p))
        print("Number of answers 1p: " + str(n_up_1p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_up))
        directory_path = filepath + "/" + "up"
        filenames = ["upup_", "up2p_", "up2u_", "up1p_"]
        datas = [[queries_up_up, answers_up_up_filters, answers_up_up],
                 [queries_up_2p, answers_up_2p_filters, answers_up_2p],
                 [queries_up_2u, answers_up_2u_filters, answers_up_2u],
                 [queries_up_1p, answers_up_1p_filters, answers_up_1p]]
        save_pkl_file(directory_path, filenames, datas)


def read_queries(dataset, query_structures, gen_num, max_ans_num, gen_train, gen_valid, gen_test, query_names,
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

    idx = 0
    query_name = query_names[idx] if save_name else str(idx)

    name_to_save = query_name
    set_logger("./data/{}/".format(dataset), name_to_save)

    if gen_valid:
        valid_queries_path = "./data/{}/valid-queries.pkl".format(dataset)
        valid_queries = read_pkl_file(valid_queries_path)
        for query_structure in query_structures:
            queries = valid_queries[list2tuple(query_structure)]
            find_answers(query_structure, queries,
                         valid_only_ent_in, valid_only_ent_out,
                         valid_ent_in, valid_ent_out,
                         train_ent_in, train_ent_out,
                         'valid', dataset)

    if gen_test:
        test_queries_path = "./data/{}/test-queries.pkl".format(dataset)
        test_queries = read_pkl_file(test_queries_path)
        for query_structure in query_structures:
            queries = test_queries[list2tuple(query_structure)]
            find_answers(query_structure, queries,
                         test_only_ent_in,
                         test_only_ent_out,
                         test_ent_in, test_ent_out,
                         valid_ent_in, valid_ent_out,
                         'test',dataset)

    idx += 1


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
@click.option('--gen_test', is_flag=True, default=True)
@click.option('--gen_id', default=0)
@click.option('--save_name', is_flag=True, default=False)
@click.option('--index_only', is_flag=True, default=False)
def main(dataset, seed, gen_train_num, gen_valid_num, gen_test_num, max_ans_num, reindex, gen_train, gen_valid,
         gen_test, gen_id, save_name, index_only):
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

    e = 'e'
    r = 'r'
    n = 'n'
    u = 'u'

    query_structures = [
        [e, [r]],
         [e, [r, r]],

        [e, [r, r, r]],

        [[e, [r]], [e, [r]]],
        [[e, [r]], [e, [r]], [e, [r]]],
        [[e, [r, r]], [e, [r]]],
        [[[e, [r]], [e, [r]]], [r]],
        # negation
        # [[e, [r]], [e, [r, n]]],
        # [[e, [r]], [e, [r]], [e, [r, n]]],
        # [[e, [r, r]], [e, [r, n]]],
        # [[e, [r, r, n]], [e, [r]]],
        # [[[e, [r]], [e, [r, n]]], [r]],
        # union
        [[e, [r]], [e, [r]], [u]],
        [[[e, [r]], [e, [r]], [u]], [r]]
    ]

    query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']

    read_queries(dataset, query_structures, [gen_train_num, gen_valid_num, gen_test_num],
                 max_ans_num, gen_train, gen_valid, gen_test, query_names[gen_id:gen_id + 1], save_name)


if __name__ == '__main__':
    main()
