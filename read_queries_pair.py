import pickle
import os.path as osp
import numpy as np
import click
from collections import defaultdict, Counter
import random
from copy import deepcopy
import time
import pdb
import logging
import os
import itertools as it

from matplotlib import pyplot as plt


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
    os.makedirs(directory_path, exist_ok=True)
    for filename,data in zip(filenames,datas):
        filename = directory_path + "/" + filename
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(data, f)

def add_query_answ_dict(query_structure, query, query_set, filter_out_set,pred_answer_set, filters, answers,cardinality_left=None,cardinality_right=None,cardinality_value=None,nele=0):
    if cardinality_left is None and len(answers) > 0:
        query_set[list2tuple(query_structure)].add(list2tuple(query))
        filter_out_set[list2tuple(query)] = filters
        pred_answer_set[list2tuple(query)] = answers
        return query_set, filter_out_set, pred_answer_set
    elif cardinality_left is None and len(answers) == 0:
        return query_set, filter_out_set, pred_answer_set

    if cardinality_left is not None and cardinality_right is not None and len(answers) > 0:

        if cardinality_left==cardinality_right:
            if cardinality_value == cardinality_left:
                query_set[list2tuple(query_structure)].add(list2tuple(query))
                filter_out_set[list2tuple(query)] = filters
                pred_answer_set[list2tuple(query)] = answers
                nele += len(answers)
        elif cardinality_value>cardinality_left and cardinality_value<=cardinality_right:
            query_set[list2tuple(query_structure)].add(list2tuple(query))
            filter_out_set[list2tuple(query)] = filters
            pred_answer_set[list2tuple(query)] = answers
            nele += len(answers)
    return query_set,filter_out_set,pred_answer_set,nele

def compute_answers_query_2p(entity,rels,ent_out1, ent_out2):
    answer_set_final =set()
    answer_set_inter = ent_out1[entity][rels[0]]
    for i, ent in enumerate(answer_set_inter):
        answer_set_final.update(ent_out2[ent][rels[1]])
    return answer_set_final

def compute_answers_query_2pn(entity,rels,ent_out1, ent_out2):
    answer_set_neg =set()
    answer_set_inter = ent_out1[entity][rels[0]]
    for i, ent in enumerate(answer_set_inter):
        answer_set_neg.update(ent_out2[ent][rels[1]])
    answer_set_final = answer_set_inter - answer_set_neg
    return answer_set_final


def compute_answers_query_3p(entity,rels,ent_out1, ent_out2, ent_out3):
    answer_set_final = set()
    answer_set_inter2 =set()
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

def find_answers(query_structure, queries, computed_hard_answer_set, missing_ent_in, missing_ent_out, all_ent_in, all_ent_out, easy_ent_in,
                 easy_ent_out, mode, dataset,):
    '''
    missing_ent = entities related only to the validation/test set
    all_ent = entities related to the train + validation/test set
    easy_ent = entities related only to train

    '''
    random.seed(0)
    num_sampled = 0
    folder_name_query_red = mode + "-query-reduction-prop"
    filepath_query_red = "./data/{}/{}".format(dataset,folder_name_query_red)
    folder_name_card = mode + "-query-card-prop"
    filepath_card = "./data/{}/{}".format(dataset, folder_name_card)
    filenames = ["test-queries", "test-easy-answers", "test-hard-answers"]
    entity_set = set(range(len(all_ent_in)))

    queries_2p_2p, queries_2p_1p, queries_2p_1p_fully = defaultdict(set), defaultdict(set), defaultdict(set)
    answers_2p_2p, answers_2p_2p_filters, answers_2p_1p, answers_2p_1p_filters, answers_2p_1p_fully, answers_2p_1p_fully_filters = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set), defaultdict(set), defaultdict(set)

    queries_1p_1p = defaultdict(set)
    answers_1p_1p, answers_1p_1p_filters = defaultdict(set), defaultdict(set)

    queries_3p_3p ,queries_3p_2p , queries_3p_1p = defaultdict(set), defaultdict(set),defaultdict(set)
    answers_3p_3p , answers_3p_3p_filters , answers_3p_2p , answers_3p_2p_filters ,answers_3p_1p , answers_3p_1p_filters= defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_4p_4p,queries_4p_3p, queries_4p_2p, queries_4p_1p = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_4p_4p, answers_4p_4p_filters, answers_4p_3p, answers_4p_3p_filters, answers_4p_2p, answers_4p_2p_filters, answers_4p_1p, answers_4p_1p_filters = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)

    queries_2i_2i , queries_2i_1p = defaultdict(set),defaultdict(set)
    answers_2i_2i , answers_2i_2i_filters , answers_2i_1p , answers_2i_1p_filters = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_3i_3i ,queries_3i_2i , queries_3i_1p = defaultdict(set),defaultdict(set),defaultdict(set)
    answers_3i_3i , answers_3i_3i_filters , answers_3i_2i , answers_3i_2i_filters , answers_3i_1p , answers_3i_1p_filters= defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_4i_4i, queries_4i_3i, queries_4i_2i, queries_4i_1p = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_4i_4i, answers_4i_4i_filters, answers_4i_3i, answers_4i_3i_filters, answers_4i_2i, answers_4i_2i_filters, answers_4i_1p, answers_4i_1p_filters = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)

    queries_pi_pi , queries_pi_2i , queries_pi_2p , queries_pi_1p= defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)
    answers_pi_pi , answers_pi_pi_filters , answers_pi_2i , answers_pi_2i_filters , answers_pi_2p , answers_pi_2p_filters , answers_pi_1p , answers_pi_1p_filters = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_ip_ip , queries_ip_2i , queries_ip_2p , queries_ip_1p = defaultdict(set), defaultdict(set),defaultdict(set),defaultdict(set)
    answers_ip_ip , answers_ip_ip_filters , answers_ip_2i , answers_ip_2i_filters , answers_ip_2p , answers_ip_2p_filters , answers_ip_1p , answers_ip_1p_filters = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_up_up , queries_up_2u , queries_up_2p , queries_up_1p = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)
    answers_up_up , answers_up_up_filters , answers_up_2u , answers_up_2u_filters , answers_up_2p , answers_up_2p_filters , answers_up_1p , answers_up_1p_filters = defaultdict(
        set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)

    queries_2u_1p, answers_2u_1p_filters, answers_2u_1p = defaultdict(set),defaultdict(set),defaultdict(set)
    queries_2u_2u, answers_2u_2u_filters, answers_2u_2u = defaultdict(set), defaultdict(set), defaultdict(set)

    queries_2in_pos_exist, queries_2in_pos_only_missing, queries_2in_neg_exist, queries_2in_neg_only_missing = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_2in_pos_exist, filtered_2in_pos_exist, answers_2in_pos_only_missing, filtered_2in_pos_only_missing, answers_2in_neg_exist, filtered_2in_neg_exist, answers_2in_neg_only_missing,filtered_2in_neg_only_missing = defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    n_2in_pos_exist = n_2in_pos_only_missing = n_2in_neg_exist = n_2in_neg_only_missing = 0

    queries_3in_pos_exist, queries_3in_pos_only_missing, queries_3in_neg_exist, queries_3in_neg_only_missing = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_3in_pos_exist, filtered_3in_pos_exist, answers_3in_pos_only_missing, filtered_3in_pos_only_missing, answers_3in_neg_exist, filtered_3in_neg_exist, answers_3in_neg_only_missing, filtered_3in_neg_only_missing = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    n_3in_pos_exist = n_3in_pos_only_missing = n_3in_neg_exist = n_3in_neg_only_missing = 0

    queries_pin_pos_exist, queries_pin_pos_only_missing, queries_pin_neg_exist, queries_pin_neg_only_missing = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_pin_pos_exist, filtered_pin_pos_exist, answers_pin_pos_only_missing, filtered_pin_pos_only_missing, answers_pin_neg_exist, filtered_pin_neg_exist, answers_pin_neg_only_missing, filtered_pin_neg_only_missing = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    n_pin_pos_exist = n_pin_pos_only_missing = n_pin_neg_exist = n_pin_neg_only_missing = 0

    queries_pni_pos_exist, queries_pni_pos_only_missing, queries_pni_neg_exist, queries_pni_neg_only_missing = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_pni_pos_exist, filtered_pni_pos_exist, answers_pni_pos_only_missing, filtered_pni_pos_only_missing, answers_pni_neg_exist, filtered_pni_neg_exist, answers_pni_neg_only_missing, filtered_pni_neg_only_missing = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    n_pni_pos_exist = n_pni_pos_only_missing = n_pni_neg_exist = n_pni_neg_only_missing = 0

    queries_inp_pos_exist, queries_inp_pos_only_missing, queries_inp_neg_exist, queries_inp_neg_only_missing = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_inp_pos_exist, filtered_inp_pos_exist, answers_inp_pos_only_missing, filtered_inp_pos_only_missing, answers_inp_neg_exist, filtered_inp_neg_exist, answers_inp_neg_only_missing, filtered_inp_neg_only_missing = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    n_inp_pos_exist = n_inp_pos_only_missing = n_inp_neg_exist = n_inp_neg_only_missing = 0


    queries_3p_reduced, answers_3p_reduced, answers_3p_filters_reduced = defaultdict(set), defaultdict(
        set), defaultdict(set)
    queries_4p_reduced, answers_4p_reduced, answers_4p_filters_reduced = defaultdict(set), defaultdict(
        set), defaultdict(set)
    queries_3i_reduced, answers_3i_reduced, answers_3i_filters_reduced = defaultdict(set), defaultdict(
        set), defaultdict(set)
    queries_4i_reduced, answers_4i_reduced, answers_4i_filters_reduced = defaultdict(set), defaultdict(
        set), defaultdict(set)
    queries_pi_reduced, answers_pi_reduced, answers_pi_filters_reduced = defaultdict(set), defaultdict(
        set), defaultdict(set)
    queries_ip_reduced, answers_ip_reduced, answers_ip_filters_reduced = defaultdict(set), defaultdict(
        set), defaultdict(set)
    queries_up_reduced, answers_up_reduced, answers_up_filters_reduced = defaultdict(set), defaultdict(
        set), defaultdict(set)
    queries_up_overall, answers_up_filters_overall, answers_up_overall = defaultdict(set), defaultdict(
        set), defaultdict(set)
    n_1p_1p = 0
    n_1p_2p = n_2p_2p = 0
    n_1p_3p = n_2p_3p = n_3p_3p = 0
    n_1p_4p = n_2p_4p = n_3p_4p = n_4p_4p = 0
    n_2i_1i = n_2i_2i = n_3i_1i = n_3i_2i = n_3i_3i = 0
    n_1p_4i = n_2i_4i = n_3i_4i = n_4i_4i = 0
    n_pi_1p = n_pi_2p = n_pi_2i = n_pi_pi = 0
    n_ip_1p = n_ip_2p = n_ip_2i = n_ip_ip = 0
    n_up_1p = n_up_2p = n_up_2u = n_up_up = n_up_0p=0
    n_2u_2u = n_2u_1p = 0
    #negation
    n_2in_2in = n_2in_1p =0

    n_tot_hard_answers_1p = n_tot_hard_answers_2p = n_tot_hard_answers_3p = n_tot_hard_answers_4p = n_tot_hard_answers_2i = n_tot_hard_answers_3i = n_tot_hard_answers_4i = n_tot_hard_answers_pi = n_tot_hard_answers_ip = n_tot_hard_answers_up = n_tot_hard_answers_2u= n_tot_hard_answers_2in = 0

    # intermediate inits
    queries_2p_0c_reduced, queries_2p_0c_reduced,queries_2p_1c_reduced, queries_2p_2c_reduced,queries_2p_10c_reduced,queries_2p_100c_reduced  = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_2p_0c_reduced, answers_2p_0c_filters_reduced,answers_2p_1c_reduced, answers_2p_1c_filters_reduced, answers_2p_2c_reduced, answers_2p_2c_filters_reduced,answers_2p_10c_reduced, answers_2p_10c_filters_reduced,answers_2p_100c_reduced, answers_2p_100c_filters_reduced = defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    n_2p_0c_reduced = n_2p_1c_reduced = n_2p_2c_reduced = n_2p_10c_reduced = n_2p_100c_reduced = 0

    queries_2p_0c_true, queries_2p_0c_true, queries_2p_1c_true, queries_2p_2c_true, queries_2p_10c_true, queries_2p_100c_true = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_2p_0c_true, answers_2p_0c_filters_true, answers_2p_1c_true, answers_2p_1c_filters_true, answers_2p_2c_true, answers_2p_2c_filters_true, answers_2p_10c_true, answers_2p_10c_filters_true, answers_2p_100c_true, answers_2p_100c_filters_true = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_2p_0c_true = n_2p_1c_true = n_2p_2c_true = n_2p_10c_true = n_2p_100c_true = 0

    queries_3p_0c_reduced, queries_3p_0c_reduced, queries_3p_1c_reduced, queries_3p_2c_reduced, queries_3p_10c_reduced, queries_3p_100c_reduced = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_3p_0c_reduced, answers_3p_0c_filters_reduced, answers_3p_1c_reduced, answers_3p_1c_filters_reduced, answers_3p_2c_reduced, answers_3p_2c_filters_reduced, answers_3p_10c_reduced, answers_3p_10c_filters_reduced, answers_3p_100c_reduced, answers_3p_100c_filters_reduced = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_3p_0c_reduced = n_3p_1c_reduced = n_3p_2c_reduced = n_3p_10c_reduced = n_3p_100c_reduced = 0

    queries_3p_0c_true, queries_3p_1c_true, queries_3p_2c_true, queries_3p_10c_true, queries_3p_100c_true = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_3p_0c_true, answers_3p_0c_filters_true, answers_3p_1c_true, answers_3p_1c_filters_true, answers_3p_2c_true, answers_3p_2c_filters_true, answers_3p_10c_true, answers_3p_10c_filters_true, answers_3p_100c_true, answers_3p_100c_filters_true = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_3p_0c_true = n_3p_1c_true = n_3p_2c_true = n_3p_10c_true = n_3p_100c_true = 0

    queries_4p_0c_reduced, queries_4p_0c_reduced, queries_4p_1c_reduced, queries_4p_2c_reduced, queries_4p_10c_reduced, queries_4p_100c_reduced = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_4p_0c_reduced, answers_4p_0c_filters_reduced, answers_4p_1c_reduced, answers_4p_1c_filters_reduced, answers_4p_2c_reduced, answers_4p_2c_filters_reduced, answers_4p_10c_reduced, answers_4p_10c_filters_reduced, answers_4p_100c_reduced, answers_4p_100c_filters_reduced = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_4p_0c_reduced = n_4p_1c_reduced = n_4p_2c_reduced = n_4p_10c_reduced = n_4p_100c_reduced = 0

    queries_4p_0c_true, queries_4p_1c_true, queries_4p_2c_true, queries_4p_10c_true, queries_4p_100c_true = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_4p_0c_true, answers_4p_0c_filters_true, answers_4p_1c_true, answers_4p_1c_filters_true, answers_4p_2c_true, answers_4p_2c_filters_true, answers_4p_10c_true, answers_4p_10c_filters_true, answers_4p_100c_true, answers_4p_100c_filters_true = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_4p_0c_true = n_4p_1c_true = n_4p_2c_true = n_4p_10c_true = n_4p_100c_true = 0



    queries_pi_0c_reduced, queries_pi_0c_reduced, queries_pi_1c_reduced, queries_pi_2c_reduced, queries_pi_10c_reduced, queries_pi_100c_reduced = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_pi_0c_reduced, answers_pi_0c_filters_reduced, answers_pi_1c_reduced, answers_pi_1c_filters_reduced, answers_pi_2c_reduced, answers_pi_2c_filters_reduced, answers_pi_10c_reduced, answers_pi_10c_filters_reduced, answers_pi_100c_reduced, answers_pi_100c_filters_reduced = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_pi_0c_reduced = n_pi_1c_reduced = n_pi_2c_reduced = n_pi_10c_reduced = n_pi_100c_reduced = 0

    queries_pi_0c_true, queries_pi_0c_true, queries_pi_1c_true, queries_pi_2c_true, queries_pi_10c_true, queries_pi_100c_true = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_pi_0c_true, answers_pi_0c_filters_true, answers_pi_1c_true, answers_pi_1c_filters_true, answers_pi_2c_true, answers_pi_2c_filters_true, answers_pi_10c_true, answers_pi_10c_filters_true, answers_pi_100c_true, answers_pi_100c_filters_true = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_pi_0c_true = n_pi_1c_true = n_pi_2c_true = n_pi_10c_true = n_pi_100c_true = 0

    queries_ip_0c_reduced, queries_ip_0c_reduced, queries_ip_1c_reduced, queries_ip_2c_reduced, queries_ip_10c_reduced, queries_ip_100c_reduced = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_ip_0c_reduced, answers_ip_0c_filters_reduced, answers_ip_1c_reduced, answers_ip_1c_filters_reduced, answers_ip_2c_reduced, answers_ip_2c_filters_reduced, answers_ip_10c_reduced, answers_ip_10c_filters_reduced, answers_ip_100c_reduced, answers_ip_100c_filters_reduced = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_ip_0c_reduced = n_ip_1c_reduced = n_ip_2c_reduced = n_ip_10c_reduced = n_ip_100c_reduced = 0

    queries_ip_0c_true, queries_ip_0c_true, queries_ip_1c_true, queries_ip_2c_true, queries_ip_10c_true, queries_ip_100c_true = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_ip_0c_true, answers_ip_0c_filters_true, answers_ip_1c_true, answers_ip_1c_filters_true, answers_ip_2c_true, answers_ip_2c_filters_true, answers_ip_10c_true, answers_ip_10c_filters_true, answers_ip_100c_true, answers_ip_100c_filters_true = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_ip_0c_true = n_ip_1c_true = n_ip_2c_true = n_ip_10c_true = n_ip_100c_true = 0

    queries_up_0c_reduced, queries_up_0c_reduced, queries_up_1c_reduced, queries_up_2c_reduced, queries_up_10c_reduced, queries_up_100c_reduced = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_up_0c_reduced, answers_up_0c_filters_reduced, answers_up_1c_reduced, answers_up_1c_filters_reduced, answers_up_2c_reduced, answers_up_2c_filters_reduced, answers_up_10c_reduced, answers_up_10c_filters_reduced, answers_up_100c_reduced, answers_up_100c_filters_reduced = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_up_0c_reduced = n_up_1c_reduced = n_up_2c_reduced = n_up_10c_reduced = n_up_100c_reduced = 0

    queries_up_0c_true, queries_up_0c_true, queries_up_1c_true, queries_up_2c_true, queries_up_10c_true, queries_up_100c_true = defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    answers_up_0c_true, answers_up_0c_filters_true, answers_up_1c_true, answers_up_1c_filters_true, answers_up_2c_true, answers_up_2c_filters_true, answers_up_10c_true, answers_up_10c_filters_true, answers_up_100c_true, answers_up_100c_filters_true = defaultdict(
        set), defaultdict(
        set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(
        set), defaultdict(set), defaultdict(set)
    n_up_0c_true = n_up_1c_true = n_up_2c_true = n_up_10c_true = n_up_100c_true = 0


    rel_per_query_1p, rel_per_query_2p, rel_per_query_3p, rel_per_query_4p, rel_per_query_2i, rel_per_query_3i, rel_per_query_4i, rel_per_query_pi, rel_per_query_ip, rel_per_query_2u, rel_per_query_up,rel_per_query_2in,rel_per_query_3in,rel_per_query_pin,rel_per_query_pni,rel_per_query_inp= {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}
    anch_per_query_1p, anch_per_query_2p, anch_per_query_3p, anch_per_query_4p,anch_per_query_2i, anch_per_query_3i, anch_per_query_4i,anch_per_query_pi, anch_per_query_ip, anch_per_query_2u, anch_per_query_up,anch_per_query_2in,anch_per_query_3in,anch_per_query_pin,anch_per_query_pni,anch_per_query_inp = {},{}, {}, {}, {}, {}, {}, {}, {},{},{}, {}, {}, {},{},{}

    n_tot_hard_answers_3in = n_3in_3in = n_3in_1p = n_3in_part_neg = n_3in_full_neg = n_2in_part_neg = n_2in_full_neg= 0
    n_tot_hard_answers_pin = n_pin_pin = n_pin_1p = n_pin_part_neg = n_pin_full_neg= 0
    n_tot_hard_answers_pni = n_pni_pni = n_pni_1p = n_pin_part_neg = n_pin_full_neg= 0
    n_tot_hard_answers_inp = n_inp_inp = n_inp_1p = n_inp_part_neg = n_inp_full_neg= 0
    new_easy_answer_set_2in, new_easy_answer_set_3in,new_easy_answer_set_pin,new_easy_answer_set_pni,new_easy_answer_set_inp  = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)
    new_hard_answer_set_2in, new_hard_answer_set_3in,new_hard_answer_set_pin,new_hard_answer_set_pni,new_hard_answer_set_inp = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)
    queries_2in, queries_3in, queries_pin, queries_pni, queries_inp = defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set),defaultdict(set)
    for query in queries:
        query = tuple2list(query)
        answer_set = achieve_answer(query, all_ent_in, all_ent_out)
        easy_answer_set = achieve_answer(query, easy_ent_in, easy_ent_out)
        hard_answer_set = computed_hard_answer_set[list2tuple(query)]
        if query_structure == ['e', ['r']]:
            n_tot_hard_answers_1p+=len(hard_answer_set)
            entity = query[0]
            rel1 = query[1][0]

            set_rel = ({rel1})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_1p:
                    rel_per_query_1p[rel] += len(hard_answer_set)
                else:
                    rel_per_query_1p[rel] = len(hard_answer_set)
            set_ent = ({entity})
            for ent in set_ent:
                if ent in anch_per_query_1p:
                    anch_per_query_1p[ent] += len(hard_answer_set)
                else:
                    anch_per_query_1p[ent] = len(hard_answer_set)

            reachable_answers_1p = missing_ent_out[entity][rel1]
            queries_1p_1p,answers_1p_1p_filters,answers_1p_1p = add_query_answ_dict(query_structure, query, queries_1p_1p, answers_1p_1p_filters, answers_1p_1p, easy_answer_set, reachable_answers_1p)
            n_1p_1p+=len(reachable_answers_1p)

        if query_structure == ['e', ['r', 'r']]:
            # 0 existing 1 predicted
            n_tot_hard_answers_2p += len(hard_answer_set)
            # check if answer reachable with training link
            entity = query[0]
            rel1 = query[1][0]
            rel2 = query[1][1]

            set_rel = ({rel1, rel2})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_2p:
                    rel_per_query_2p[rel] += len(hard_answer_set)
                else:
                    rel_per_query_2p[rel] = len(hard_answer_set)
            set_ent = ({entity})
            for ent in set_ent:
                if ent in anch_per_query_2p:
                    anch_per_query_2p[ent] += len(hard_answer_set)
                else:
                    anch_per_query_2p[ent] = len(hard_answer_set)

            n_intermediate_ext_answers = len(easy_ent_out[entity][rel1])
            reachableanswers01 = compute_answers_query_2p(entity, [rel1,rel2], easy_ent_out, missing_ent_out)
            reachableanswers10 = compute_answers_query_2p(entity, [rel1, rel2], missing_ent_out, easy_ent_out)
            reachable_answers_1p = ((reachableanswers01|reachableanswers10) - easy_answer_set) & hard_answer_set
            n_1p_2p += len(reachable_answers_1p)

            reachable_answers_2p = set()
            if len(reachable_answers_1p) < len(hard_answer_set):
                reachableanswers11 = compute_answers_query_2p(entity, [rel1, rel2], missing_ent_out, missing_ent_out)
                reachable_answers_2p = (reachableanswers11 - reachable_answers_1p - easy_answer_set) & hard_answer_set

                n_2p_2p += len(reachable_answers_2p)

            ##cardinality 'true'
            queries_2p_1p, answers_2p_1p_filters, answers_2p_1p = add_query_answ_dict(query_structure, query,
                                                                                      queries_2p_1p,
                                                                                      answers_2p_1p_filters,
                                                                                      answers_2p_1p,
                                                                                      answer_set - reachable_answers_1p,
                                                                                      reachable_answers_1p)
            queries_2p_2p, answers_2p_2p_filters, answers_2p_2p = add_query_answ_dict(query_structure, query,
                                                                                      queries_2p_2p,
                                                                                      answers_2p_2p_filters,
                                                                                      answers_2p_2p,
                                                                                      answer_set - reachable_answers_2p,
                                                                                      reachable_answers_2p)
            #cardinality
            queries_2p_0c_true,answers_2p_0c_filters_true,answers_2p_0c_true,n_2p_0c_true =add_query_answ_dict(query_structure,
                                                                                                  query, queries_2p_0c_true,
                                                                                                  answers_2p_0c_filters_true,
                                                                                                  answers_2p_0c_true,
                                                                                                  answer_set-reachable_answers_2p,
                                                                                                  reachable_answers_2p,
                                                                                                  0,0,n_intermediate_ext_answers,n_2p_0c_true)
            queries_2p_1c_true, answers_2p_1c_filters_true, answers_2p_1c_true,n_2p_1c_true = add_query_answ_dict(query_structure,
                                                                                                     query,
                                                                                                     queries_2p_1c_true,
                                                                                                     answers_2p_1c_filters_true,
                                                                                                     answers_2p_1c_true,
                                                                                                     answer_set - reachable_answers_2p,
                                                                                                     reachable_answers_2p,
                                                                                                     1, 1,
                                                                                                     n_intermediate_ext_answers,n_2p_1c_true)
            queries_2p_2c_true, answers_2p_2c_filters_true, answers_2p_2c_true,n_2p_2c_true = add_query_answ_dict(query_structure,
                                                                                                     query,
                                                                                                     queries_2p_2c_true,
                                                                                                     answers_2p_2c_filters_true,
                                                                                                     answers_2p_2c_true,
                                                                                                     answer_set - reachable_answers_2p,
                                                                                                     reachable_answers_2p,
                                                                                                     1, 9,
                                                                                                     n_intermediate_ext_answers,n_2p_2c_true)
            queries_2p_10c_true, answers_2p_10c_filters_true, answers_2p_10c_true,n_2p_10c_true = add_query_answ_dict(query_structure,
                                                                                                     query,
                                                                                                     queries_2p_10c_true,
                                                                                                     answers_2p_10c_filters_true,
                                                                                                     answers_2p_10c_true,
                                                                                                     answer_set - reachable_answers_2p,
                                                                                                     reachable_answers_2p,
                                                                                                     9, 99,
                                                                                                     n_intermediate_ext_answers,n_2p_10c_true)
            queries_2p_100c_true, answers_2p_100c_filters_true, answers_2p_100c_true,n_2p_100c_true = add_query_answ_dict(query_structure,
                                                                                                     query,
                                                                                                     queries_2p_100c_true,
                                                                                                     answers_2p_100c_filters_true,
                                                                                                     answers_2p_100c_true,
                                                                                                     answer_set - reachable_answers_2p,
                                                                                                     reachable_answers_2p,
                                                                                                     99, 10000000000000000,
                                                                                                     n_intermediate_ext_answers,n_2p_100c_true)
            queries_2p_0c_reduced, answers_2p_0c_filters_reduced, answers_2p_0c_reduced,n_2p_0c_reduced = add_query_answ_dict(query_structure,
                                                                                                     query,
                                                                                                     queries_2p_0c_reduced,
                                                                                                     answers_2p_0c_filters_reduced,
                                                                                                     answers_2p_0c_reduced,
                                                                                                     answer_set - reachable_answers_1p,
                                                                                                     reachable_answers_1p,
                                                                                                     0, 0,
                                                                                                     n_intermediate_ext_answers,n_2p_0c_reduced)
            queries_2p_1c_reduced, answers_2p_1c_filters_reduced, answers_2p_1c_reduced,n_2p_1c_reduced = add_query_answ_dict(query_structure,
                                                                                                     query,
                                                                                                     queries_2p_1c_reduced,
                                                                                                     answers_2p_1c_filters_reduced,
                                                                                                     answers_2p_1c_reduced,
                                                                                                     answer_set - reachable_answers_1p,
                                                                                                     reachable_answers_1p,
                                                                                                     1, 1,
                                                                                                     n_intermediate_ext_answers,n_2p_1c_reduced)
            queries_2p_2c_reduced, answers_2p_2c_filters_reduced, answers_2p_2c_reduced,n_2p_2c_reduced = add_query_answ_dict(query_structure,
                                                                                                     query,
                                                                                                     queries_2p_2c_reduced,
                                                                                                     answers_2p_2c_filters_reduced,
                                                                                                     answers_2p_2c_reduced,
                                                                                                     answer_set - reachable_answers_1p,
                                                                                                     reachable_answers_1p,
                                                                                                     1, 9,
                                                                                                     n_intermediate_ext_answers,n_2p_2c_reduced)
            queries_2p_10c_reduced, answers_2p_10c_filters_reduced, answers_2p_10c_reduced,n_2p_10c_reduced = add_query_answ_dict(query_structure,
                                                                                                        query,
                                                                                                        queries_2p_10c_reduced,
                                                                                                        answers_2p_10c_filters_reduced,
                                                                                                        answers_2p_10c_reduced,
                                                                                                        answer_set - reachable_answers_1p,
                                                                                                        reachable_answers_1p,
                                                                                                        9, 99,
                                                                                                        n_intermediate_ext_answers,n_2p_10c_reduced)
            queries_2p_100c_reduced, answers_2p_100c_filters_reduced, answers_2p_100c_reduced,n_2p_100c_reduced = add_query_answ_dict(
                                                                                                        query_structure,
                                                                                                        query,
                                                                                                        queries_2p_100c_reduced,
                                                                                                        answers_2p_100c_filters_reduced,
                                                                                                        answers_2p_100c_reduced,
                                                                                                        answer_set - reachable_answers_1p,
                                                                                                        reachable_answers_1p,
                                                                                                        99, 10000000000000000,
                                                                                                        n_intermediate_ext_answers,n_2p_100c_reduced)
        if query_structure == ['e', ['r', 'r', 'r']]:
            # 0 existing 1 predicted
            n_tot_hard_answers_3p += len(hard_answer_set)
            
            reachable_answers_2p = set()
            reachable_answers_3p = set()

            entity = query[0]
            rel1 = query[1][0]
            rel2 = query[1][1]
            rel3 = query[1][2]

            set_rel = ({rel1, rel2, rel3})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_3p:
                    rel_per_query_3p[rel] += len(hard_answer_set)
                else:
                    rel_per_query_3p[rel] = len(hard_answer_set)
            set_ent = ({entity})
            for ent in set_ent:
                if ent in anch_per_query_3p:
                    anch_per_query_3p[ent] += len(hard_answer_set)
                else:
                    anch_per_query_3p[ent] = len(hard_answer_set)

            # take the max among the two variables
            var1 = easy_ent_out[entity][rel1]
            n_intermediate_ext_answers= len(var1)
            for ele in var1:
                if len(easy_ent_out[ele][rel2])>n_intermediate_ext_answers:
                    n_intermediate_ext_answers = len(easy_ent_out[ele][rel2])
            # existing + existing + predicted --> 1p 001
            reachable_answers_001 = compute_answers_query_3p(entity, [rel1, rel2, rel3], easy_ent_out, easy_ent_out, missing_ent_out)
            # existing + predicted + existing --> 1p 010
            reachable_answers_010 = compute_answers_query_3p(entity, [rel1, rel2, rel3], easy_ent_out, missing_ent_out, easy_ent_out)
            # predicted + existing + existing --> 1p 100
            reachable_answers_100 = compute_answers_query_3p(entity, [rel1, rel2, rel3],missing_ent_out,easy_ent_out, easy_ent_out)

            reachable_answers_1p = ((reachable_answers_001 | reachable_answers_010 | reachable_answers_100) - easy_answer_set) & hard_answer_set  # subtract the easy answers
            n_1p_3p += len(reachable_answers_1p)

            if len(reachable_answers_1p) < len(hard_answer_set):
                # continue the computation for 2p/3p
                # existing + predicted + predicted --> 2p 011
                reachable_answers_011 = compute_answers_query_3p(entity, [rel1, rel2, rel3], easy_ent_out,
                                                                 missing_ent_out, missing_ent_out)
                # predicted + predicted + existing --> 2p 110
                reachable_answers_110 = compute_answers_query_3p(entity, [rel1, rel2, rel3], missing_ent_out,
                                                                 missing_ent_out, easy_ent_out)
                # predicted + existing + predicted --> 2p 101
                reachable_answers_101 = compute_answers_query_3p(entity, [rel1, rel2, rel3], missing_ent_out,
                                                                 easy_ent_out, missing_ent_out)
                reachable_answers_2p = ((reachable_answers_011 | reachable_answers_110 | reachable_answers_101) - reachable_answers_1p - easy_answer_set) & hard_answer_set  # subtract the easy answers and the 1p answers
                n_2p_3p += len(reachable_answers_2p)
                if len(reachable_answers_1p | reachable_answers_2p) < len(hard_answer_set):
                    # predicted + predicted + existing --> 3p 111
                    reachable_answers_111 = compute_answers_query_3p(entity, [rel1, rel2, rel3], missing_ent_out,
                                                                     missing_ent_out, missing_ent_out)
                    reachable_answers_3p = (reachable_answers_111 - reachable_answers_2p - reachable_answers_1p - easy_answer_set) & hard_answer_set  # subtract the easy answers and the 1p/2p answers
                    n_3p_3p += len(reachable_answers_3p)

            queries_3p_1p, answers_3p_1p_filters, answers_3p_1p = add_query_answ_dict(query_structure, query,
                                                                                          queries_3p_1p,
                                                                                          answers_3p_1p_filters,
                                                                                          answers_3p_1p,
                                                                                          answer_set - reachable_answers_1p,
                                                                                         reachable_answers_1p)
            queries_3p_2p, answers_3p_2p_filters, answers_3p_2p = add_query_answ_dict(query_structure, query,
                                                                                      queries_3p_2p,
                                                                                      answers_3p_2p_filters,
                                                                                      answers_3p_2p,
                                                                                      answer_set - reachable_answers_2p,
                                                                                      reachable_answers_2p)
            queries_3p_3p, answers_3p_3p_filters, answers_3p_3p = add_query_answ_dict(query_structure, query,
                                                                                      queries_3p_3p,
                                                                                      answers_3p_3p_filters,
                                                                                      answers_3p_3p,
                                                                                      answer_set - reachable_answers_3p,
                                                                                      reachable_answers_3p)
            # cardinality
            queries_3p_0c_true, answers_3p_0c_filters_true, answers_3p_0c_true, n_3p_0c_true = add_query_answ_dict(
                query_structure,
                query, queries_3p_0c_true,
                answers_3p_0c_filters_true,
                answers_3p_0c_true,
                answer_set - reachable_answers_3p,
                reachable_answers_3p,
                0, 0, n_intermediate_ext_answers, n_3p_0c_true)
            queries_3p_1c_true, answers_3p_1c_filters_true, answers_3p_1c_true, n_3p_1c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_3p_1c_true,
                answers_3p_1c_filters_true,
                answers_3p_1c_true,
                answer_set - reachable_answers_3p,
                reachable_answers_3p,
                1, 1,
                n_intermediate_ext_answers, n_3p_1c_true)
            queries_3p_2c_true, answers_3p_2c_filters_true, answers_3p_2c_true, n_3p_2c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_3p_2c_true,
                answers_3p_2c_filters_true,
                answers_3p_2c_true,
                answer_set - reachable_answers_3p,
                reachable_answers_3p,
                1, 9,
                n_intermediate_ext_answers, n_3p_2c_true)
            queries_3p_10c_true, answers_3p_10c_filters_true, answers_3p_10c_true, n_3p_10c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_3p_10c_true,
                answers_3p_10c_filters_true,
                answers_3p_10c_true,
                answer_set - reachable_answers_3p,
                reachable_answers_3p,
                9, 99,
                n_intermediate_ext_answers, n_3p_10c_true)
            queries_3p_100c_true, answers_3p_100c_filters_true, answers_3p_100c_true, n_3p_100c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_3p_100c_true,
                answers_3p_100c_filters_true,
                answers_3p_100c_true,
                answer_set - reachable_answers_3p,
                reachable_answers_3p,
                99, 10000000000000000,
                n_intermediate_ext_answers, n_3p_100c_true)
            queries_3p_0c_reduced, answers_3p_0c_filters_reduced, answers_3p_0c_reduced, n_3p_0c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_3p_0c_reduced,
                answers_3p_0c_filters_reduced,
                answers_3p_0c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p,
                reachable_answers_1p|reachable_answers_2p,
                0, 0,
                n_intermediate_ext_answers, n_3p_0c_reduced)
            queries_3p_1c_reduced, answers_3p_1c_filters_reduced, answers_3p_1c_reduced, n_3p_1c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_3p_1c_reduced,
                answers_3p_1c_filters_reduced,
                answers_3p_1c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p,
                reachable_answers_1p | reachable_answers_2p,
                1, 1,
                n_intermediate_ext_answers, n_3p_1c_reduced)
            queries_3p_2c_reduced, answers_3p_2c_filters_reduced, answers_3p_2c_reduced, n_3p_2c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_3p_2c_reduced,
                answers_3p_2c_filters_reduced,
                answers_3p_2c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p,
                reachable_answers_1p | reachable_answers_2p,
                1, 9,
                n_intermediate_ext_answers, n_3p_2c_reduced)
            queries_3p_10c_reduced, answers_3p_10c_filters_reduced, answers_3p_10c_reduced, n_3p_10c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_3p_10c_reduced,
                answers_3p_10c_filters_reduced,
                answers_3p_10c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p,
                reachable_answers_1p | reachable_answers_2p,
                9, 99,
                n_intermediate_ext_answers, n_3p_10c_reduced)
            queries_3p_100c_reduced, answers_3p_100c_filters_reduced, answers_3p_100c_reduced, n_3p_100c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_3p_100c_reduced,
                answers_3p_100c_filters_reduced,
                answers_3p_100c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p,
                reachable_answers_1p | reachable_answers_2p,
                99, 10000000000000000,
                n_intermediate_ext_answers, n_3p_100c_reduced)
            queries_3p_reduced, answers_3p_filters_reduced, answers_3p_reduced= add_query_answ_dict(
                query_structure,
                query,
                queries_3p_reduced,
                answers_3p_filters_reduced,
                answers_3p_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p,
                reachable_answers_1p | reachable_answers_2p)
        if query_structure == ['e', ['r', 'r', 'r', 'r']]:
            reachable_answers_4p = set()
            n_tot_hard_answers_4p+=len(hard_answer_set)
            entity = query[0]
            rel1 = query[1][0]
            rel2 = query[1][1]
            rel3 = query[1][2]
            rel4 = query[1][3]

            set_rel = ({rel1,rel2,rel3,rel4})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
               if rel in rel_per_query_4p:
                   rel_per_query_4p[rel] += len(hard_answer_set)
               else:
                   rel_per_query_4p[rel] = len(hard_answer_set)

            if entity in anch_per_query_4p:
                anch_per_query_4p[entity] += len(hard_answer_set)
            else:
                anch_per_query_4p[entity] = len(hard_answer_set)

            var1 = easy_ent_out[entity][rel1]
            n_intermediate_ext_answers = len(var1)
            for ele in var1:
                var2 = easy_ent_out[ele][rel2]
                if len(var2) > n_intermediate_ext_answers:
                    n_intermediate_ext_answers = len(easy_ent_out[ele][rel2])
                for ele2 in var2:
                    var3 = easy_ent_out[ele][rel3]
                    if len(var3) > n_intermediate_ext_answers:
                        n_intermediate_ext_answers = len(easy_ent_out[ele2][rel3])

            reachable_answers_2p, reachable_answers_3p = set(), set()
            # 1p 0001
            reachable_answers_0001 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], easy_ent_out,
                                                              easy_ent_out, easy_ent_out, missing_ent_out)
            # 0010
            reachable_answers_0010 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], easy_ent_out,
                                                              easy_ent_out,
                                                              missing_ent_out, easy_ent_out)
            # 0100
            reachable_answers_0100 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], easy_ent_out,
                                                              missing_ent_out,
                                                              easy_ent_out, easy_ent_out)
            # 1000
            reachable_answers_1000 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], missing_ent_out,
                                                              easy_ent_out,
                                                              easy_ent_out, easy_ent_out)
            reachable_answers_1p = ((reachable_answers_0001 | reachable_answers_0010 | reachable_answers_0100 | reachable_answers_1000) - easy_answer_set) & hard_answer_set  # subtract the train answers
            n_1p_4p+=len(reachable_answers_1p)
            if len(reachable_answers_1p) < len(hard_answer_set):
                # 2p
                # 0011
                reachable_answers_0011 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], easy_ent_out,
                                                                  easy_ent_out, missing_ent_out, missing_ent_out)
                # 0101
                reachable_answers_0101 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], easy_ent_out,
                                                                  missing_ent_out, easy_ent_out, missing_ent_out)
                # 0110
                reachable_answers_0110 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], easy_ent_out,
                                                                  missing_ent_out, missing_ent_out, easy_ent_out)
                # 1001
                reachable_answers_1001 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], missing_ent_out,
                                                                  easy_ent_out, easy_ent_out, missing_ent_out)
                # 1010
                reachable_answers_1010 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], missing_ent_out,
                                                                  easy_ent_out, missing_ent_out, easy_ent_out)
                # 1100
                reachable_answers_1100 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], missing_ent_out,
                                                                  missing_ent_out, easy_ent_out, easy_ent_out)

                reachable_answers_2p = ((reachable_answers_0011 | reachable_answers_0101 | reachable_answers_0110 | reachable_answers_1001 | reachable_answers_1010 | reachable_answers_1100) - reachable_answers_1p - easy_answer_set) & hard_answer_set  # subtract the train answers and the 1p answers
                n_2p_4p += len(reachable_answers_2p)
                if len(reachable_answers_1p | reachable_answers_2p) < len(hard_answer_set):
                    # 3p
                    # 0111
                    reachable_answers_0111 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4], easy_ent_out,
                                                                      missing_ent_out, missing_ent_out,
                                                                      missing_ent_out)
                    # 1011
                    reachable_answers_1011 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4],
                                                                      missing_ent_out,
                                                                      easy_ent_out, missing_ent_out,
                                                                      missing_ent_out)
                    # 1101
                    reachable_answers_1101 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4],
                                                                      missing_ent_out,
                                                                      missing_ent_out, easy_ent_out,
                                                                      missing_ent_out)
                    # 1110
                    reachable_answers_1110 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4],
                                                                      missing_ent_out,
                                                                      missing_ent_out, missing_ent_out,
                                                                      easy_ent_out)
                    reachable_answers_3p = ((reachable_answers_0111 | reachable_answers_1011 | reachable_answers_1101 | reachable_answers_1110) - reachable_answers_2p - reachable_answers_1p - easy_answer_set) & hard_answer_set  # subtract the train answers and the 1p/2p answers
                    n_3p_4p += len(reachable_answers_3p)
                    if len(reachable_answers_1p | reachable_answers_2p | reachable_answers_3p) < len(hard_answer_set):
                        reachable_answers_1111 = compute_answers_query_4p(entity, [rel1, rel2, rel3, rel4],
                                                                          missing_ent_out,
                                                                          missing_ent_out, missing_ent_out,
                                                                          missing_ent_out)

                        reachable_answers_4p = (reachable_answers_1111 - reachable_answers_3p - reachable_answers_2p - reachable_answers_1p - easy_answer_set) & hard_answer_set  # subtract the train answers and the 1p/2p answers
                        n_4p_4p += len(reachable_answers_4p)

            queries_4p_1p, answers_4p_1p_filters, answers_4p_1p = add_query_answ_dict(query_structure, query,
                                                                                      queries_4p_1p,
                                                                                      answers_4p_1p_filters,
                                                                                      answers_4p_1p,
                                                                                      answer_set - reachable_answers_1p,
                                                                                      reachable_answers_1p)
            queries_4p_2p, answers_4p_2p_filters, answers_4p_2p = add_query_answ_dict(query_structure, query,
                                                                                      queries_4p_2p,
                                                                                      answers_4p_2p_filters,
                                                                                      answers_4p_2p,
                                                                                      answer_set - reachable_answers_2p,
                                                                                      reachable_answers_2p)
            queries_4p_3p, answers_4p_3p_filters, answers_4p_3p = add_query_answ_dict(query_structure, query,
                                                                                      queries_4p_3p,
                                                                                      answers_4p_3p_filters,
                                                                                      answers_4p_3p,
                                                                                      answer_set - reachable_answers_3p,
                                                                                      reachable_answers_3p)
            queries_4p_4p, answers_4p_4p_filters, answers_4p_4p = add_query_answ_dict(query_structure, query,
                                                                                      queries_4p_4p,
                                                                                      answers_4p_4p_filters,
                                                                                      answers_4p_4p,
                                                                                      answer_set - reachable_answers_4p,
                                                                                      reachable_answers_4p)
            # cardinality
            queries_4p_0c_true, answers_4p_0c_filters_true, answers_4p_0c_true, n_4p_0c_true = add_query_answ_dict(
                query_structure,
                query, queries_4p_0c_true,
                answers_4p_0c_filters_true,
                answers_4p_0c_true,
                answer_set - reachable_answers_4p,
                reachable_answers_4p,
                0, 0, n_intermediate_ext_answers, n_4p_0c_true)
            queries_4p_1c_true, answers_4p_1c_filters_true, answers_4p_1c_true, n_4p_1c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_4p_1c_true,
                answers_4p_1c_filters_true,
                answers_4p_1c_true,
                answer_set - reachable_answers_4p,
                reachable_answers_4p,
                1, 1,
                n_intermediate_ext_answers, n_4p_1c_true)
            queries_4p_2c_true, answers_4p_2c_filters_true, answers_4p_2c_true, n_4p_2c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_4p_2c_true,
                answers_4p_2c_filters_true,
                answers_4p_2c_true,
                answer_set - reachable_answers_4p,
                reachable_answers_4p,
                1, 9,
                n_intermediate_ext_answers, n_4p_2c_true)
            queries_4p_10c_true, answers_4p_10c_filters_true, answers_4p_10c_true, n_4p_10c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_4p_10c_true,
                answers_4p_10c_filters_true,
                answers_4p_10c_true,
                answer_set - reachable_answers_4p,
                reachable_answers_4p,
                9, 99,
                n_intermediate_ext_answers, n_4p_10c_true)
            queries_4p_100c_true, answers_4p_100c_filters_true, answers_4p_100c_true, n_4p_100c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_4p_100c_true,
                answers_4p_100c_filters_true,
                answers_4p_100c_true,
                answer_set - reachable_answers_4p,
                reachable_answers_4p,
                99, 10000000000000000,
                n_intermediate_ext_answers, n_4p_100c_true)

            queries_4p_reduced, answers_4p_filters_reduced, answers_4p_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_4p_reduced,
                answers_4p_filters_reduced,
                answers_4p_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_3p,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_3p)

            queries_4p_0c_reduced, answers_4p_0c_filters_reduced, answers_4p_0c_reduced, n_4p_0c_reduced = add_query_answ_dict(
                query_structure,
                query, queries_4p_0c_reduced,
                answers_4p_0c_filters_reduced,
                answers_4p_0c_reduced,
                answer_set - (reachable_answers_1p | reachable_answers_2p | reachable_answers_3p),
                (reachable_answers_1p | reachable_answers_2p | reachable_answers_3p),
                0, 0, n_intermediate_ext_answers, n_4p_0c_reduced)
            queries_4p_1c_reduced, answers_4p_1c_filters_reduced, answers_4p_1c_reduced, n_4p_1c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_4p_1c_reduced,
                answers_4p_1c_filters_reduced,
                answers_4p_1c_reduced,
                answer_set - (reachable_answers_1p | reachable_answers_2p | reachable_answers_3p),
                (reachable_answers_1p | reachable_answers_2p | reachable_answers_3p),
                1, 1,
                n_intermediate_ext_answers, n_4p_1c_reduced)
            queries_4p_2c_reduced, answers_4p_2c_filters_reduced, answers_4p_2c_reduced, n_4p_2c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_4p_2c_reduced,
                answers_4p_2c_filters_reduced,
                answers_4p_2c_reduced,
                answer_set - (reachable_answers_1p | reachable_answers_2p | reachable_answers_3p),
                (reachable_answers_1p | reachable_answers_2p | reachable_answers_3p),
                1, 9,
                n_intermediate_ext_answers, n_4p_2c_reduced)
            queries_4p_10c_reduced, answers_4p_10c_filters_reduced, answers_4p_10c_reduced, n_4p_10c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_4p_10c_reduced,
                answers_4p_10c_filters_reduced,
                answers_4p_10c_reduced,
                answer_set - (reachable_answers_1p | reachable_answers_2p | reachable_answers_3p),
                (reachable_answers_1p | reachable_answers_2p | reachable_answers_3p),
                9, 99,
                n_intermediate_ext_answers, n_4p_10c_reduced)
            queries_4p_100c_reduced, answers_4p_100c_filters_reduced, answers_4p_100c_reduced, n_4p_100c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_4p_100c_reduced,
                answers_4p_100c_filters_reduced,
                answers_4p_100c_reduced,
                answer_set - (reachable_answers_1p | reachable_answers_2p | reachable_answers_3p),
                (reachable_answers_1p | reachable_answers_2p | reachable_answers_3p),
                99, 10000000000000000,
                n_intermediate_ext_answers, n_4p_100c_reduced)



        if query_structure == [['e', ['r']], ['e', ['r']]]:
            n_tot_hard_answers_2i += len(hard_answer_set)
            reachable_answers_2i = set()
            # 2i
            entity1 = query[0][0]
            rel1 = query[0][1][0]
            entity2 = query[1][0]
            rel2 = query[1][1][0]

            set_rel = ({rel1, rel2})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_2i:
                    rel_per_query_2i[rel] += len(hard_answer_set)
                else:
                    rel_per_query_2i[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2})
            for ent in set_ent:
                if ent in anch_per_query_2i:
                    anch_per_query_2i[ent] += len(hard_answer_set)
                else:
                    anch_per_query_2i[ent] = len(hard_answer_set)

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

            reachable_answers_1p = ((answers_01 | answers_10) - easy_answer_set) & hard_answer_set
            n_2i_1i += len(reachable_answers_1p)

            if len(reachable_answers_1p) < len(hard_answer_set):
                # 11
                # compute the answers of the query (entity1,rel1,?y) on the missing graph
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                # compute the answers of the query (entity2,rel2,?y) on the missing graph
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answers_11 = answer_set_q1_1 & answer_set_q1_2
                reachable_answers_2i = (answers_11 - reachable_answers_1p - easy_answer_set) & hard_answer_set
                n_2i_2i += len(reachable_answers_2i)
            queries_2i_2i, answers_2i_2i_filters, answers_2i_2i = add_query_answ_dict(query_structure, query,
                                                                                      queries_2i_2i,
                                                                                      answers_2i_2i_filters,
                                                                                      answers_2i_2i,
                                                                                      answer_set - reachable_answers_2i,
                                                                                      reachable_answers_2i)
            queries_2i_1p, answers_2i_1p_filters, answers_2i_1p = add_query_answ_dict(query_structure, query,
                                                                                      queries_2i_1p,
                                                                                      answers_2i_1p_filters,
                                                                                      answers_2i_1p,
                                                                                      answer_set - reachable_answers_1p,
                                                                                      reachable_answers_1p)
        if query_structure == [['e', ['r']], ['e', ['r']], ['e', ['r']]]:
            n_tot_hard_answers_3i += len(hard_answer_set)
            reachable_answers_2i = set()
            reachable_answers_3i = set()
            # 3i
            entity1 = query[0][0]
            rel1 = query[0][1][0]
            entity2 = query[1][0]
            rel2 = query[1][1][0]
            entity3 = query[2][0]
            rel3 = query[2][1][0]

            set_rel = ({rel1, rel2, rel3})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_3i:
                    rel_per_query_3i[rel] += len(hard_answer_set)
                else:
                    rel_per_query_3i[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2, entity3})
            for ent in set_ent:
                if ent in anch_per_query_3i:
                    anch_per_query_3i[ent] += len(hard_answer_set)
                else:
                    anch_per_query_3i[ent] = len(hard_answer_set)

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
            reachable_answers_1p = ((answers_001 | answers_010 | answers_100) - easy_answer_set) & hard_answer_set
            n_3i_1i += len(reachable_answers_1p)

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

                reachable_answers_2i = ((answers_011 | answers_101 | answers_110) - reachable_answers_1p - easy_answer_set) & hard_answer_set
                n_3i_2i += len(reachable_answers_2i)

                if len(reachable_answers_2i | reachable_answers_1p) < len(hard_answer_set):
                    # 111
                    answer_set_q1_1 = missing_ent_out[entity1][rel1]
                    answer_set_q1_2 = missing_ent_out[entity2][rel2]
                    answer_set_q1_3 = missing_ent_out[entity3][rel3]
                    answers_111 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3

                    reachable_answers_3i = ((answers_111 - reachable_answers_1p - reachable_answers_2i) - easy_answer_set) & hard_answer_set
                    n_3i_3i += len(reachable_answers_3i)
            queries_3i_1p, answers_3i_1p_filters, answers_3i_1p = add_query_answ_dict(query_structure, query,
                                                                                      queries_3i_1p,
                                                                                      answers_3i_1p_filters,
                                                                                      answers_3i_1p,
                                                                                      answer_set - reachable_answers_1p,
                                                                                      reachable_answers_1p)
            queries_3i_2i, answers_3i_2i_filters, answers_3i_2i = add_query_answ_dict(query_structure, query,
                                                                                      queries_3i_2i,
                                                                                      answers_3i_2i_filters,
                                                                                      answers_3i_2i,
                                                                                      answer_set - reachable_answers_2i,
                                                                                      reachable_answers_2i)
            queries_3i_3i, answers_3i_3i_filters, answers_3i_3i = add_query_answ_dict(query_structure, query,
                                                                                          queries_3i_3i,
                                                                                          answers_3i_3i_filters,
                                                                                          answers_3i_3i,
                                                                                          answer_set - reachable_answers_3i,
                                                                                          reachable_answers_3i)

            queries_3i_reduced, answers_3i_filters_reduced, answers_3i_reduced  = add_query_answ_dict(
                query_structure,
                query,
                queries_3i_reduced,
                answers_3i_filters_reduced,
                answers_3i_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2i)
        if query_structure == [['e', ['r']], ['e', ['r']], ['e', ['r']], ['e', ['r']]]:
            # 4i
            reachable_answers_4i = set()
            n_tot_hard_answers_4i += len(hard_answer_set)
            entity1 = query[0][0]
            rel1 = query[0][1][0]
            entity2 = query[1][0]
            rel2 = query[1][1][0]
            entity3 = query[2][0]
            rel3 = query[2][1][0]
            entity4 = query[3][0]
            rel4 = query[3][1][0]
            set_rel = ({rel1, rel2, rel3, rel4})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_4i:
                    rel_per_query_4i[rel] += len(hard_answer_set)
                else:
                    rel_per_query_4i[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2, entity3, entity4})
            for ent in set_ent:
                if ent in anch_per_query_4i:
                    anch_per_query_4i[ent] += len(hard_answer_set)
                else:
                    anch_per_query_4i[ent] = len(hard_answer_set)

            reachable_answers_2i = set()
            reachable_answers_3i = set()
            # 0001
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answer_set_q1_3 = easy_ent_out[entity3][rel3]
            answer_set_q1_4 = missing_ent_out[entity4][rel4]
            answers_0001 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4
            # 0010
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answer_set_q1_3 = missing_ent_out[entity3][rel3]
            answer_set_q1_4 = easy_ent_out[entity4][rel4]
            answers_0010 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4
            # 0100
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = missing_ent_out[entity2][rel2]
            answer_set_q1_3 = easy_ent_out[entity3][rel3]
            answer_set_q1_4 = easy_ent_out[entity4][rel4]
            answers_0100 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

            # 1000
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answer_set_q1_3 = easy_ent_out[entity3][rel3]
            answer_set_q1_4 = easy_ent_out[entity4][rel4]
            answers_1000 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

            reachable_answers_1p = ((answers_0001 | answers_0010 | answers_0100 | answers_1000) - easy_answer_set) & hard_answer_set
            n_1p_4i+=len(reachable_answers_1p)

            if len(reachable_answers_1p) < len(hard_answer_set):

                # 0011
                answer_set_q1_1 = easy_ent_out[entity1][rel1]
                answer_set_q1_2 = easy_ent_out[entity2][rel2]
                answer_set_q1_3 = missing_ent_out[entity3][rel3]
                answer_set_q1_4 = missing_ent_out[entity4][rel4]
                answers_0011 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                # 0101
                answer_set_q1_1 = easy_ent_out[entity1][rel1]
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answer_set_q1_3 = easy_ent_out[entity3][rel3]
                answer_set_q1_4 = missing_ent_out[entity4][rel4]
                answers_0101 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                # 0110
                answer_set_q1_1 = easy_ent_out[entity1][rel1]
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answer_set_q1_3 = missing_ent_out[entity3][rel3]
                answer_set_q1_4 = easy_ent_out[entity4][rel4]
                answers_0110 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                # 1001
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                answer_set_q1_2 = easy_ent_out[entity2][rel2]
                answer_set_q1_3 = easy_ent_out[entity3][rel3]
                answer_set_q1_4 = missing_ent_out[entity4][rel4]
                answers_1001 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                # 1010
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                answer_set_q1_2 = easy_ent_out[entity2][rel2]
                answer_set_q1_3 = missing_ent_out[entity3][rel3]
                answer_set_q1_4 = easy_ent_out[entity4][rel4]
                answers_1010 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                # 1100
                answer_set_q1_1 = missing_ent_out[entity1][rel1]
                answer_set_q1_2 = missing_ent_out[entity2][rel2]
                answer_set_q1_3 = easy_ent_out[entity3][rel3]
                answer_set_q1_4 = easy_ent_out[entity4][rel4]
                answers_1100 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                reachable_answers_2i = ((answers_0011 | answers_0101 | answers_0110 | answers_1001 | answers_1010 | answers_1100) - reachable_answers_1p - easy_answer_set) & hard_answer_set
                n_2i_4i += len(reachable_answers_2i)
                if len(reachable_answers_2i | reachable_answers_1p) < len(hard_answer_set):

                    # 0111
                    answer_set_q1_1 = easy_ent_out[entity1][rel1]
                    answer_set_q1_2 = missing_ent_out[entity2][rel2]
                    answer_set_q1_3 = missing_ent_out[entity3][rel3]
                    answer_set_q1_4 = missing_ent_out[entity4][rel4]
                    answers_0111 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                    # 1011
                    answer_set_q1_1 = missing_ent_out[entity1][rel1]
                    answer_set_q1_2 = easy_ent_out[entity2][rel2]
                    answer_set_q1_3 = missing_ent_out[entity3][rel3]
                    answer_set_q1_4 = missing_ent_out[entity4][rel4]
                    answers_1011 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4
                    # 1101
                    answer_set_q1_1 = missing_ent_out[entity1][rel1]
                    answer_set_q1_2 = missing_ent_out[entity2][rel2]
                    answer_set_q1_3 = easy_ent_out[entity3][rel3]
                    answer_set_q1_4 = missing_ent_out[entity4][rel4]
                    answers_1101 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4
                    # 1110
                    answer_set_q1_1 = missing_ent_out[entity1][rel1]
                    answer_set_q1_2 = missing_ent_out[entity2][rel2]
                    answer_set_q1_3 = missing_ent_out[entity3][rel3]
                    answer_set_q1_4 = easy_ent_out[entity4][rel4]
                    answers_1110 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4

                    reachable_answers_3i = ((answers_0111 | answers_1011 | answers_1101 | answers_1110) - reachable_answers_1p - reachable_answers_2i - easy_answer_set) & hard_answer_set
                    n_3i_4i += len(reachable_answers_3i)
                    if len(reachable_answers_3i | reachable_answers_2i | reachable_answers_1p) < len(hard_answer_set):
                        # 1111
                        answer_set_q1_1 = missing_ent_out[entity1][rel1]
                        answer_set_q1_2 = missing_ent_out[entity2][rel2]
                        answer_set_q1_3 = missing_ent_out[entity3][rel3]
                        answer_set_q1_4 = missing_ent_out[entity4][rel4]
                        answers_1111 = answer_set_q1_1 & answer_set_q1_2 & answer_set_q1_3 & answer_set_q1_4
                        reachable_answers_4i = (answers_1111 - reachable_answers_1p - reachable_answers_2i - reachable_answers_3i - easy_answer_set) & hard_answer_set
                        n_4i_4i += len(reachable_answers_4i)
            queries_4i_1p, answers_4i_1p_filters, answers_4i_1p = add_query_answ_dict(query_structure, query,
                                                                                      queries_4i_1p,
                                                                                      answers_4i_1p_filters,
                                                                                      answers_4i_1p,
                                                                                      answer_set - reachable_answers_1p,
                                                                                      reachable_answers_1p)
            queries_4i_2i, answers_4i_2i_filters, answers_4i_2i = add_query_answ_dict(query_structure, query,
                                                                                      queries_4i_2i,
                                                                                      answers_4i_2i_filters,
                                                                                      answers_4i_2i,
                                                                                      answer_set - reachable_answers_2i,
                                                                                      reachable_answers_2i)
            queries_4i_3i, answers_4i_3i_filters, answers_4i_3i = add_query_answ_dict(query_structure, query,
                                                                                      queries_4i_3i,
                                                                                      answers_4i_3i_filters,
                                                                                      answers_4i_3i,
                                                                                      answer_set - reachable_answers_3i,
                                                                                      reachable_answers_3i)
            queries_4i_4i, answers_4i_4i_filters, answers_4i_4i = add_query_answ_dict(query_structure, query,
                                                                                      queries_4i_4i,
                                                                                      answers_4i_4i_filters,
                                                                                      answers_4i_4i,
                                                                                      answer_set - reachable_answers_4i,
                                                                                      reachable_answers_4i)
            queries_4i_reduced, answers_4i_filters_reduced, answers_4i_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_4i_reduced,
                answers_4i_filters_reduced,
                answers_4i_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2i - reachable_answers_3i,
                reachable_answers_1p | reachable_answers_2i | reachable_answers_3i)
        if query_structure == [['e', ['r', 'r']], ['e', ['r']]]:
            # ((e1,r1.?x) and (?x,r2,?y))and(e2,r3,?y)
            reachable_answers_2i = set()
            reachable_answers_2p = set()
            reachable_answers_pi = set()
            n_tot_hard_answers_pi += len(hard_answer_set)
            entity1 = query[0][0]
            rel1 = query[0][1][0]
            rel2 = query[0][1][1]
            entity2 = query[1][0]
            rel3 = query[1][1][0]

            set_rel = ({rel1, rel2, rel3})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_pi:
                    rel_per_query_pi[rel] += len(hard_answer_set)
                else:
                    rel_per_query_pi[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2})
            for ent in set_ent:
                if ent in anch_per_query_pi:
                    anch_per_query_pi[ent] += len(hard_answer_set)
                else:
                    anch_per_query_pi[ent] = len(hard_answer_set)
            n_intermediate_ext_answers = len(easy_ent_out[entity1][rel1])
            # 001
            subquery_2p_answers_00 = compute_answers_query_2p(entity1,[rel1,rel2],easy_ent_out, easy_ent_out)

            subquery_1p_answers_1 = missing_ent_out[entity2][rel3]
            answers_001 = subquery_2p_answers_00 & subquery_1p_answers_1
            # 010
            subquery_2p_answers_01 = compute_answers_query_2p(entity1, [rel1, rel2], easy_ent_out, missing_ent_out)
            subquery_1p_answers_0 = easy_ent_out[entity2][rel3]
            answers_010 = subquery_2p_answers_01 & subquery_1p_answers_0
            # 100
            subquery_2p_answers_10 = compute_answers_query_2p(entity1, [rel1, rel2], missing_ent_out, easy_ent_out)
            answers_100 = subquery_2p_answers_10 & subquery_1p_answers_0
            reachable_answers_1p = ((answers_001 | answers_010 | answers_100) - easy_answer_set) & hard_answer_set
            n_pi_1p += len(reachable_answers_1p)

            if len(reachable_answers_1p) < len(hard_answer_set):
                # 2i
                # 011
                answers_011 = subquery_2p_answers_01 & subquery_1p_answers_1
                # 101
                answers_101 = subquery_2p_answers_10 & subquery_1p_answers_1
                reachable_answers_2i = ((answers_011 | answers_101) - reachable_answers_1p - easy_answer_set) & hard_answer_set
                n_pi_2i += len(reachable_answers_2i)

                # 2p
                # 110
                subquery_2p_answers_11 = compute_answers_query_2p(entity1, [rel1, rel2], missing_ent_out, missing_ent_out)
                answers_110 = subquery_2p_answers_11 & subquery_1p_answers_0
                reachable_answers_2p = (answers_110 - reachable_answers_1p - reachable_answers_2i - easy_answer_set) & hard_answer_set
                n_pi_2p += len(reachable_answers_2p)

                if len(reachable_answers_2p | reachable_answers_2i | reachable_answers_1p) < len(hard_answer_set):
                    # 111
                    # pi
                    answers_111 = subquery_2p_answers_11 & subquery_1p_answers_1
                    reachable_answers_pi = (answers_111 - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i - easy_answer_set) & hard_answer_set
                    n_pi_pi += len(reachable_answers_pi)
                    #n_intermediate_answerspi.append(len(subquery_2p_answers_00))
            queries_pi_1p, answers_pi_1p_filters, answers_pi_1p = add_query_answ_dict(query_structure, query,
                                                                                      queries_pi_1p,
                                                                                      answers_pi_1p_filters,
                                                                                      answers_pi_1p,
                                                                                      answer_set - reachable_answers_1p,
                                                                                      reachable_answers_1p)
            queries_pi_2i, answers_pi_2i_filters, answers_pi_2i = add_query_answ_dict(query_structure, query,
                                                                                      queries_pi_2i,
                                                                                      answers_pi_2i_filters,
                                                                                      answers_pi_2i,
                                                                                      answer_set - reachable_answers_2i,
                                                                                      reachable_answers_2i)
            queries_pi_2p, answers_pi_2p_filters, answers_pi_2p = add_query_answ_dict(query_structure, query,
                                                                                      queries_pi_2p,
                                                                                      answers_pi_2p_filters,
                                                                                      answers_pi_2p,
                                                                                      answer_set - reachable_answers_2p,
                                                                                      reachable_answers_2p)
            queries_pi_pi, answers_pi_pi_filters, answers_pi_pi = add_query_answ_dict(query_structure, query,
                                                                                  queries_pi_pi,
                                                                                  answers_pi_pi_filters,
                                                                                  answers_pi_pi,
                                                                                  answer_set - reachable_answers_pi,
                                                                                  reachable_answers_pi)
            # cardinality
            queries_pi_0c_true, answers_pi_0c_filters_true, answers_pi_0c_true, n_pi_0c_true = add_query_answ_dict(
                query_structure,
                query, queries_pi_0c_true,
                answers_pi_0c_filters_true,
                answers_pi_0c_true,
                answer_set - reachable_answers_pi,
                reachable_answers_pi,
                0, 0, n_intermediate_ext_answers, n_pi_0c_true)
            queries_pi_1c_true, answers_pi_1c_filters_true, answers_pi_1c_true, n_pi_1c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_pi_1c_true,
                answers_pi_1c_filters_true,
                answers_pi_1c_true,
                answer_set - reachable_answers_pi,
                reachable_answers_pi,
                1, 1,
                n_intermediate_ext_answers, n_pi_1c_true)
            queries_pi_2c_true, answers_pi_2c_filters_true, answers_pi_2c_true, n_pi_2c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_pi_2c_true,
                answers_pi_2c_filters_true,
                answers_pi_2c_true,
                answer_set - reachable_answers_pi,
                reachable_answers_pi,
                1, 9,
                n_intermediate_ext_answers, n_pi_2c_true)
            queries_pi_10c_true, answers_pi_10c_filters_true, answers_pi_10c_true, n_pi_10c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_pi_10c_true,
                answers_pi_10c_filters_true,
                answers_pi_10c_true,
                answer_set - reachable_answers_pi,
                reachable_answers_pi,
                9, 99,
                n_intermediate_ext_answers, n_pi_10c_true)
            queries_pi_100c_true, answers_pi_100c_filters_true, answers_pi_100c_true, n_pi_100c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_pi_100c_true,
                answers_pi_100c_filters_true,
                answers_pi_100c_true,
                answer_set - reachable_answers_pi,
                reachable_answers_pi,
                99, 10000000000000000,
                n_intermediate_ext_answers, n_pi_100c_true)
            queries_pi_0c_reduced, answers_pi_0c_filters_reduced, answers_pi_0c_reduced, n_pi_0c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_pi_0c_reduced,
                answers_pi_0c_filters_reduced,
                answers_pi_0c_reduced,
                answer_set - reachable_answers_1p-reachable_answers_2p-reachable_answers_2i,
                reachable_answers_1p|reachable_answers_2p|reachable_answers_2i,
                0, 0,
                n_intermediate_ext_answers, n_pi_0c_reduced)
            queries_pi_1c_reduced, answers_pi_1c_filters_reduced, answers_pi_1c_reduced, n_pi_1c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_pi_1c_reduced,
                answers_pi_1c_filters_reduced,
                answers_pi_1c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i,
                1, 1,
                n_intermediate_ext_answers, n_pi_1c_reduced)
            queries_pi_2c_reduced, answers_pi_2c_filters_reduced, answers_pi_2c_reduced, n_pi_2c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_pi_2c_reduced,
                answers_pi_2c_filters_reduced,
                answers_pi_2c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i,
                1, 9,
                n_intermediate_ext_answers, n_pi_2c_reduced)
            queries_pi_10c_reduced, answers_pi_10c_filters_reduced, answers_pi_10c_reduced, n_pi_10c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_pi_10c_reduced,
                answers_pi_10c_filters_reduced,
                answers_pi_10c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i,
                9, 99,
                n_intermediate_ext_answers, n_pi_10c_reduced)
            queries_pi_100c_reduced, answers_pi_100c_filters_reduced, answers_pi_100c_reduced, n_pi_100c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_pi_100c_reduced,
                answers_pi_100c_filters_reduced,
                answers_pi_100c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i,
                99, 10000000000000000,
                n_intermediate_ext_answers, n_pi_100c_reduced)
            queries_pi_reduced, answers_pi_filters_reduced, answers_pi_reduced  = add_query_answ_dict(
                query_structure,
                query,
                queries_pi_reduced,
                answers_pi_filters_reduced,
                answers_pi_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i)
        if query_structure == [[['e', ['r']], ['e', ['r']]], ['r']]:
            # (e1,r1,?x)and(e2,r2,?x)and(?x,r3,?y)
            n_tot_hard_answers_ip += len(hard_answer_set)
            reachable_answers_2i = set()
            reachable_answers_2p = set()
            reachable_answers_ip = set()
            entity1 = query[0][0][0]
            rel1 = query[0][0][1][0]
            entity2 = query[0][1][0]
            rel2 = query[0][1][1][0]
            rel3 = query[1][0]

            set_rel = ({rel1, rel2, rel3})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_ip:
                    rel_per_query_ip[rel] += len(hard_answer_set)
                else:
                    rel_per_query_ip[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2})
            for ent in set_ent:
                if ent in anch_per_query_ip:
                    anch_per_query_ip[ent] += len(hard_answer_set)
                else:
                    anch_per_query_ip[ent] = len(hard_answer_set)
            # 001
            answers_001 = set()
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]

            answers_00 = answer_set_q1_1 & answer_set_q1_2
            n_intermediate_ext_answers = len(answers_00)
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
            reachable_answers_1p = ((answers_001 | answers_010 | answers_100) - easy_answer_set) & hard_answer_set
            n_ip_1p += len(reachable_answers_1p)

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
                reachable_answers_2i = (answers_110 - reachable_answers_1p - easy_answer_set) & hard_answer_set
                n_ip_2i += len(reachable_answers_2i)
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
                reachable_answers_2p = ((answers_101 | answers_011) - reachable_answers_1p - reachable_answers_2i - easy_answer_set) & hard_answer_set
                n_ip_2p += len(reachable_answers_2p)

                if len(reachable_answers_2p | reachable_answers_2i | reachable_answers_1p) < len(hard_answer_set):
                    # 111
                    answers_111 = set()
                    answer_set_q1_1 = missing_ent_out[entity1][rel1]
                    answer_set_q1_2 = missing_ent_out[entity2][rel2]
                    answers_11 = answer_set_q1_1 & answer_set_q1_2
                    for ele in answers_11:
                        answers_111.update(missing_ent_out[ele][rel3])
                    reachable_answers_ip = (answers_111 - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i - easy_answer_set) & hard_answer_set
                    n_ip_ip += len(reachable_answers_ip)
                    #if len(reachable_answers_1p) >0 and len(reachable_answers_1p) <30 and len(reachable_answers_2p)>0 and len(reachable_answers_2i)>0 and len(reachable_answers_ip)>0:
                    #    print()

            queries_ip_1p, answers_ip_1p_filters, answers_ip_1p = add_query_answ_dict(query_structure, query,
                                                                                      queries_ip_1p,
                                                                                      answers_ip_1p_filters,
                                                                                      answers_ip_1p,
                                                                                      answer_set - reachable_answers_1p,
                                                                                      reachable_answers_1p)
            queries_ip_2i, answers_ip_2i_filters, answers_ip_2i = add_query_answ_dict(query_structure, query,
                                                                                      queries_ip_2i,
                                                                                      answers_ip_2i_filters,
                                                                                      answers_ip_2i,
                                                                                      answer_set - reachable_answers_2i,
                                                                                      reachable_answers_2i)
            queries_ip_2p, answers_ip_2p_filters, answers_ip_2p = add_query_answ_dict(query_structure, query,
                                                                                      queries_ip_2p,
                                                                                      answers_ip_2p_filters,
                                                                                      answers_ip_2p,
                                                                                      answer_set - reachable_answers_2p,
                                                                                      reachable_answers_2p)
            queries_ip_ip, answers_ip_ip_filters, answers_ip_ip = add_query_answ_dict(query_structure, query,
                                                                                      queries_ip_ip,
                                                                                      answers_ip_ip_filters,
                                                                                      answers_ip_ip,
                                                                                      answer_set - reachable_answers_ip,
                                                                                      reachable_answers_ip)
            # cardinality
            queries_ip_0c_true, answers_ip_0c_filters_true, answers_ip_0c_true, n_ip_0c_true = add_query_answ_dict(
                query_structure,
                query, queries_ip_0c_true,
                answers_ip_0c_filters_true,
                answers_ip_0c_true,
                answer_set - reachable_answers_ip,
                reachable_answers_ip,
                0, 0, n_intermediate_ext_answers, n_ip_0c_true)
            queries_ip_1c_true, answers_ip_1c_filters_true, answers_ip_1c_true, n_ip_1c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_ip_1c_true,
                answers_ip_1c_filters_true,
                answers_ip_1c_true,
                answer_set - reachable_answers_ip,
                reachable_answers_ip,
                1, 1,
                n_intermediate_ext_answers, n_ip_1c_true)
            queries_ip_2c_true, answers_ip_2c_filters_true, answers_ip_2c_true, n_ip_2c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_ip_2c_true,
                answers_ip_2c_filters_true,
                answers_ip_2c_true,
                answer_set - reachable_answers_ip,
                reachable_answers_ip,
                1, 9,
                n_intermediate_ext_answers, n_ip_2c_true)
            queries_ip_10c_true, answers_ip_10c_filters_true, answers_ip_10c_true, n_ip_10c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_ip_10c_true,
                answers_ip_10c_filters_true,
                answers_ip_10c_true,
                answer_set - reachable_answers_ip,
                reachable_answers_ip,
                9, 99,
                n_intermediate_ext_answers, n_ip_10c_true)
            queries_ip_100c_true, answers_ip_100c_filters_true, answers_ip_100c_true, n_ip_100c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_ip_100c_true,
                answers_ip_100c_filters_true,
                answers_ip_100c_true,
                answer_set - reachable_answers_ip,
                reachable_answers_ip,
                99, 10000000000000000,
                n_intermediate_ext_answers, n_ip_100c_true)
            queries_ip_0c_reduced, answers_ip_0c_filters_reduced, answers_ip_0c_reduced, n_ip_0c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_ip_0c_reduced,
                answers_ip_0c_filters_reduced,
                answers_ip_0c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i,
                0, 0,
                n_intermediate_ext_answers, n_ip_0c_reduced)
            queries_ip_1c_reduced, answers_ip_1c_filters_reduced, answers_ip_1c_reduced, n_ip_1c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_ip_1c_reduced,
                answers_ip_1c_filters_reduced,
                answers_ip_1c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i,
                1, 1,
                n_intermediate_ext_answers, n_ip_1c_reduced)
            queries_ip_2c_reduced, answers_ip_2c_filters_reduced, answers_ip_2c_reduced, n_ip_2c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_ip_2c_reduced,
                answers_ip_2c_filters_reduced,
                answers_ip_2c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i,
                1, 9,
                n_intermediate_ext_answers, n_ip_2c_reduced)
            queries_ip_10c_reduced, answers_ip_10c_filters_reduced, answers_ip_10c_reduced, n_ip_10c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_ip_10c_reduced,
                answers_ip_10c_filters_reduced,
                answers_ip_10c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i,
                9, 99,
                n_intermediate_ext_answers, n_ip_10c_reduced)
            queries_ip_100c_reduced, answers_ip_100c_filters_reduced, answers_ip_100c_reduced, n_ip_100c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_ip_100c_reduced,
                answers_ip_100c_filters_reduced,
                answers_ip_100c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i,
                99, 10000000000000000,
                n_intermediate_ext_answers, n_ip_100c_reduced)
            queries_ip_reduced, answers_ip_filters_reduced, answers_ip_reduced  = add_query_answ_dict(
                query_structure,
                query,
                queries_ip_reduced,
                answers_ip_filters_reduced,
                answers_ip_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2i,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2i)
        if query_structure == [[['e', ['r']], ['e', ['r']], ['u']], ['r']]:
            reachable_answers_2p = set()

            entity1 = query[0][0][0]
            rel1 = query[0][0][1][0]
            entity2 = query[0][1][0]
            rel2 = query[0][1][1][0]
            rel3 = query[1][0]
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
            reachable_answers_0p = ((answers_010 | answers_100) -easy_answer_set) & hard_answer_set # should be zero
            n_up_0p += len(reachable_answers_0p)


            # 001 #1p reduced
            answers_001 = set()
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            n_intermediate_ext_answers = len(answer_set_q1_1 | answer_set_q1_2)
            answers_00 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_00:
                answers_001.update(missing_ent_out[ele][rel3])

            answers_101 = set()
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answers_00 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_00:
                answers_101.update(missing_ent_out[ele][rel3])

            answers_011 = set()
            answer_set_q1_1 = easy_ent_out[entity1][rel1]
            answer_set_q1_2 = missing_ent_out[entity2][rel2]
            answers_00 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_00:
                answers_011.update(missing_ent_out[ele][rel3])


            reachable_answers_1p = ((answers_001 | answers_101 | answers_011) - reachable_answers_0p -easy_answer_set) & hard_answer_set
            n_up_1p += len(reachable_answers_1p)

            answers_110 = set()
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            answer_set_q1_2 = missing_ent_out[entity2][rel2]
            answers_11 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_11:
                answers_110.update(easy_ent_out[ele][rel3])
            reachable_answers_2u = (answers_110 - reachable_answers_1p- reachable_answers_0p -easy_answer_set) & hard_answer_set

            n_up_2u += len(reachable_answers_2u)
            # 111 #up reduced
            answers_111 = set()
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            answer_set_q1_2 = missing_ent_out[entity2][rel2]
            answers_11 = answer_set_q1_1 & answer_set_q1_2
            for ele in answers_11:
                answers_111.update(missing_ent_out[ele][rel3])
            reachable_answers_up = (answers_111 - reachable_answers_1p - reachable_answers_2u -reachable_answers_0p - easy_answer_set) & hard_answer_set
            n_up_up+=len(reachable_answers_up)
            queries_up_1p, answers_up_1p_filters, answers_up_1p = add_query_answ_dict(query_structure, query,
                                                                                      queries_up_1p,
                                                                                      answers_up_1p_filters,
                                                                                      answers_up_1p,
                                                                                      answer_set - reachable_answers_1p,
                                                                                      reachable_answers_1p)
            queries_up_2u, answers_up_2u_filters, answers_up_2u = add_query_answ_dict(query_structure, query,
                                                                                      queries_up_2u,
                                                                                      answers_up_2u_filters,
                                                                                      answers_up_2u,
                                                                                      answer_set - reachable_answers_2u,
                                                                                      reachable_answers_2u)
            queries_up_2p, answers_up_2p_filters, answers_up_2p = add_query_answ_dict(query_structure, query,
                                                                                      queries_up_2p,
                                                                                      answers_up_2p_filters,
                                                                                      answers_up_2p,
                                                                                      answer_set - reachable_answers_2p,
                                                                                      reachable_answers_2p)
            queries_up_up, answers_up_up_filters, answers_up_up = add_query_answ_dict(query_structure, query,
                                                                                      queries_up_up,
                                                                                      answers_up_up_filters,
                                                                                      answers_up_up,
                                                                                      answer_set - reachable_answers_up,
                                                                                      reachable_answers_up)
            # cardinality
            queries_up_0c_true, answers_up_0c_filters_true, answers_up_0c_true, n_up_0c_true = add_query_answ_dict(
                query_structure,
                query, queries_up_0c_true,
                answers_up_0c_filters_true,
                answers_up_0c_true,
                answer_set - reachable_answers_up,
                reachable_answers_up,
                0, 0, n_intermediate_ext_answers, n_up_0c_true)
            queries_up_1c_true, answers_up_1c_filters_true, answers_up_1c_true, n_up_1c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_up_1c_true,
                answers_up_1c_filters_true,
                answers_up_1c_true,
                answer_set - reachable_answers_up,
                reachable_answers_up,
                1, 1,
                n_intermediate_ext_answers, n_up_1c_true)
            queries_up_2c_true, answers_up_2c_filters_true, answers_up_2c_true, n_up_2c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_up_2c_true,
                answers_up_2c_filters_true,
                answers_up_2c_true,
                answer_set - reachable_answers_up,
                reachable_answers_up,
                1, 9,
                n_intermediate_ext_answers, n_up_2c_true)
            queries_up_10c_true, answers_up_10c_filters_true, answers_up_10c_true, n_up_10c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_up_10c_true,
                answers_up_10c_filters_true,
                answers_up_10c_true,
                answer_set - reachable_answers_up,
                reachable_answers_up,
                9, 99,
                n_intermediate_ext_answers, n_up_10c_true)
            queries_up_100c_true, answers_up_100c_filters_true, answers_up_100c_true, n_up_100c_true = add_query_answ_dict(
                query_structure,
                query,
                queries_up_100c_true,
                answers_up_100c_filters_true,
                answers_up_100c_true,
                answer_set - reachable_answers_up,
                reachable_answers_up,
                99, 10000000000000000,
                n_intermediate_ext_answers, n_up_100c_true)
            queries_up_0c_reduced, answers_up_0c_filters_reduced, answers_up_0c_reduced, n_up_0c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_up_0c_reduced,
                answers_up_0c_filters_reduced,
                answers_up_0c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2u,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2u,
                0, 0,
                n_intermediate_ext_answers, n_up_0c_reduced)
            queries_up_1c_reduced, answers_up_1c_filters_reduced, answers_up_1c_reduced, n_up_1c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_up_1c_reduced,
                answers_up_1c_filters_reduced,
                answers_up_1c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2u,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2u,
                1, 1,
                n_intermediate_ext_answers, n_up_1c_reduced)
            queries_up_2c_reduced, answers_up_2c_filters_reduced, answers_up_2c_reduced, n_up_2c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_up_2c_reduced,
                answers_up_2c_filters_reduced,
                answers_up_2c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2u,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2u,
                1, 9,
                n_intermediate_ext_answers, n_up_2c_reduced)
            queries_up_10c_reduced, answers_up_10c_filters_reduced, answers_up_10c_reduced, n_up_10c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_up_10c_reduced,
                answers_up_10c_filters_reduced,
                answers_up_10c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2u,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2u,
                9, 99,
                n_intermediate_ext_answers, n_up_10c_reduced)
            queries_up_100c_reduced, answers_up_100c_filters_reduced, answers_up_100c_reduced, n_up_100c_reduced = add_query_answ_dict(
                query_structure,
                query,
                queries_up_100c_reduced,
                answers_up_100c_filters_reduced,
                answers_up_100c_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2u,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2u,
                99, 10000000000000000,
                n_intermediate_ext_answers, n_up_100c_reduced)
            queries_up_reduced, answers_up_filters_reduced, answers_up_reduced  = add_query_answ_dict(
                query_structure,
                query,
                queries_up_reduced,
                answers_up_filters_reduced,
                answers_up_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2u,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2u)
            queries_up_overall, answers_up_filters_overall, answers_up_overall = add_query_answ_dict(
                query_structure,
                query,
                queries_up_reduced,
                answers_up_filters_reduced,
                answers_up_reduced,
                answer_set - reachable_answers_1p - reachable_answers_2p - reachable_answers_2u - reachable_answers_up,
                reachable_answers_1p | reachable_answers_2p | reachable_answers_2u | reachable_answers_up)

            n_tot_hard_answers_up += len(hard_answer_set)

            set_rel = ({rel1, rel2, rel3})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_up:
                    rel_per_query_up[rel] += len(hard_answer_set)
                else:
                    rel_per_query_up[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2})
            for ent in set_ent:
                if ent in anch_per_query_up:
                    anch_per_query_up[ent] += len(hard_answer_set)
                else:
                    anch_per_query_up[ent] = len(hard_answer_set)

        if query_structure == [['e', ['r']], ['e', ['r']], ['u']]:
            entity1 = query[0][0]
            rel1 = query[0][1][0]
            entity2 = query[1][0]
            rel2 = query[1][1][0]


            # 01
            # compute the answers of the query (entity1,rel1,?y) on the training graph
            answer_set_q1_1x = all_ent_out[entity1][rel1]
            answer_set_q1_2x = all_ent_out[entity2][rel2]

            answer_set_q1_1 =easy_ent_out[entity1][rel1]
            # compute the answers of the query (entity2,rel2,?y) on the missing graph
            answer_set_q1_2 = missing_ent_out[entity2][rel2]
            answers_01 = answer_set_q1_1 & answer_set_q1_2
            # 10
            # compute the answers of the query (entity1,rel1,?y) on the missing graph
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            # compute the answers of the query (entity2,rel2,?y) on the training graph
            answer_set_q1_2 = easy_ent_out[entity2][rel2]
            answers_10 = answer_set_q1_1 & answer_set_q1_2

            reachable_answers_1p = ((answers_01 | answers_10) - easy_answer_set) & hard_answer_set
            n_2u_1p += len(reachable_answers_1p)
            answer_set_q1_1 = missing_ent_out[entity1][rel1]
            # compute the answers of the query (entity2,rel2,?y) on the training graph
            answer_set_q1_2 = missing_ent_out[entity2][rel2]
            answers_11 = answer_set_q1_1 & answer_set_q1_2
            reachable_answers_2u = (answers_11 - reachable_answers_1p - easy_answer_set) & hard_answer_set
            n_tot_hard_answers_2u += len(hard_answer_set)
            n_2u_2u += len(reachable_answers_2u)
            queries_2u_1p, answers_2u_1p_filters, answers_2u_1p = add_query_answ_dict(query_structure, query,
                                                                                      queries_up_1p,
                                                                                      answers_up_1p_filters,
                                                                                      answers_up_1p,
                                                                                      answer_set - reachable_answers_1p,
                                                                                      reachable_answers_1p)
            queries_2u_2u, answers_2u_2u_filters, answers_2u_2u = add_query_answ_dict(query_structure, query,
                                                                                      queries_up_2u,
                                                                                      answers_up_2u_filters,
                                                                                      answers_up_2u,
                                                                                      answer_set - reachable_answers_2u,
                                                                                      reachable_answers_2u)
            set_rel = ({rel1, rel2})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_2u:
                    rel_per_query_2u[rel] += len(hard_answer_set)
                else:
                    rel_per_query_2u[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2})
            for ent in set_ent:
                if ent in anch_per_query_2u:
                    anch_per_query_2u[ent] += len(hard_answer_set)
                else:
                    anch_per_query_2u[ent] = len(hard_answer_set)


        # Negation
        #2in
        if query_structure == [['e', ['r']], ['e', ['r', 'n']]]:
            n_tot_hard_answers_2in += len(hard_answer_set)
            entity1 = query[0][0]
            rel1 = query[0][1][0] #positive
            entity2 = query[1][0]
            rel2 = query[1][1][0] #negative

            set_rel = ({rel1,rel2})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_2in:
                    rel_per_query_2in[rel] += len(hard_answer_set)
                else:
                    rel_per_query_2in[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2})
            for ent in set_ent:
                if ent in anch_per_query_2in:
                    anch_per_query_2in[ent] += len(hard_answer_set)
                else:
                    anch_per_query_2in[ent] = len(hard_answer_set)

            #x=all
            all_ent_r1 = all_ent_out[entity1][rel1]
            all_ent_r2 = all_ent_out[entity2][rel2]
            train_ent_r1 = easy_ent_out[entity1][rel1]
            train_ent_r2 = easy_ent_out[entity2][rel2]
            missing_ent_r1 = missing_ent_out[entity1][rel1]

            # easy answer set
            new_easy_answer_set = (train_ent_r1 - train_ent_r2) & answer_set
            new_hard_answer_set = answer_set - new_easy_answer_set
            answers0x = train_ent_r1 - all_ent_r2
            answers1x = missing_ent_r1 - all_ent_r2
            reachable_2in_pos_exist = (answers0x - new_easy_answer_set) & hard_answer_set
            reachable_2in_pos_only_missing = (answers1x - reachable_2in_pos_exist - new_easy_answer_set) & hard_answer_set
            n_2in_pos_only_missing += len(reachable_2in_pos_only_missing)
            n_2in_pos_exist += len(reachable_2in_pos_exist)
            queries_2in, new_easy_answer_set_2in, new_hard_answer_set_2in = add_query_answ_dict(query_structure,
                                                                                                       query,
                                                                                                       queries_2in,
                                                                                                       new_easy_answer_set_2in,
                                                                                                       new_hard_answer_set_2in,
                                                                                                       new_easy_answer_set,
                                                                                                       new_hard_answer_set)

            queries_2in_pos_exist, filtered_2in_pos_exist, answers_2in_pos_exist = add_query_answ_dict(query_structure,
                                                                                                       query,
                                                                                                       queries_2in_pos_exist,
                                                                                                       filtered_2in_pos_exist,
                                                                                                       answers_2in_pos_exist,
                                                                                                       answer_set - reachable_2in_pos_exist,
                                                                                                       reachable_2in_pos_exist)
            queries_2in_pos_only_missing, filtered_2in_pos_only_missing, answers_2in_pos_only_missing = add_query_answ_dict(
                query_structure,
                query,
                queries_2in_pos_only_missing,
                filtered_2in_pos_only_missing,
                answers_2in_pos_only_missing,
                answer_set - reachable_2in_pos_only_missing,
                reachable_2in_pos_only_missing)
        #3in
        if query_structure == [['e', ['r']],['e', ['r']], ['e', ['r', 'n']]]:

            entity1 = query[0][0]
            rel1 = query[0][1][0] # positive
            entity2 = query[1][0]
            rel2 = query[1][1][0] # positive
            entity3 = query[2][0]
            rel3 = query[2][1][0]  # negative

            set_rel = ({rel1, rel2, rel3})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_3in:
                    rel_per_query_3in[rel] += len(hard_answer_set)
                else:
                    rel_per_query_3in[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2, entity3})
            for ent in set_ent:
                if ent in anch_per_query_3in:
                    anch_per_query_3in[ent] += len(hard_answer_set)
                else:
                    anch_per_query_3in[ent] = len(hard_answer_set)


            all_ent_r3 = all_ent_out[entity3][rel3]
            train_ent_r1 = easy_ent_out[entity1][rel1]
            train_ent_r2 = easy_ent_out[entity2][rel2]
            train_ent_r3 = easy_ent_out[entity3][rel3]
            missing_ent_r1 = missing_ent_out[entity1][rel1]
            missing_ent_r2 = missing_ent_out[entity2][rel2]
            missing_ent_r3 = missing_ent_out[entity3][rel3]

            # easy answer set
            new_easy_answer_set = easy_answer_set & answer_set
            new_hard_answer_set = answer_set - new_easy_answer_set
            answers01x = (train_ent_r1 & missing_ent_r2) - all_ent_r3
            answers10x = (missing_ent_r1 & train_ent_r2) - all_ent_r3
            answers11x = (missing_ent_r1 & missing_ent_r2) - all_ent_r3
            reachable_3in_pos_exist = ((answers01x | answers10x) - new_easy_answer_set) & hard_answer_set
            reachable_3in_pos_only_missing = (answers11x - reachable_3in_pos_exist - new_easy_answer_set) & hard_answer_set
            n_3in_pos_only_missing += len(reachable_3in_pos_only_missing)
            n_3in_pos_exist += len(reachable_3in_pos_exist)

            n_tot_hard_answers_3in += len(hard_answer_set)

            queries_3in, new_easy_answer_set_3in, new_hard_answer_set_3in = add_query_answ_dict(query_structure,
                                                                                                query,
                                                                                                queries_3in,
                                                                                                new_easy_answer_set_3in,
                                                                                                new_hard_answer_set_3in,
                                                                                                new_easy_answer_set,
                                                                                                new_hard_answer_set)

            queries_3in_pos_exist, filtered_3in_pos_exist, answers_3in_pos_exist = add_query_answ_dict(query_structure,
                                                                                                       query,
                                                                                                       queries_3in_pos_exist,
                                                                                                       filtered_3in_pos_exist,
                                                                                                       answers_3in_pos_exist,
                                                                                                       answer_set - reachable_3in_pos_exist,
                                                                                                       reachable_3in_pos_exist)
            queries_3in_pos_only_missing, filtered_3in_pos_only_missing, answers_3in_pos_only_missing = add_query_answ_dict(
                query_structure,
                query,
                queries_3in_pos_only_missing,
                filtered_3in_pos_only_missing,
                answers_3in_pos_only_missing,
                answer_set - reachable_3in_pos_only_missing,
                reachable_3in_pos_only_missing)
        #pin
        if query_structure == [['e', ['r', 'r']], ['e', ['r', 'n']]]:
            n_tot_hard_answers_pin += len(hard_answer_set)
            entity1 = query[0][0]
            rel1 = query[0][1][0]
            rel2 = query[0][1][1]
            entity2 = query[1][0]
            rel3 = query[1][1][0] #negation

            set_rel = ({rel1, rel2, rel3})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_pin:
                    rel_per_query_pin[rel] += len(hard_answer_set)
                else:
                    rel_per_query_pin[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2})
            for ent in set_ent:
                if ent in anch_per_query_pin:
                    anch_per_query_pin[ent] += len(hard_answer_set)
                else:
                    anch_per_query_pin[ent] = len(hard_answer_set)

            all_ent_r3 = all_ent_out[entity2][rel3]

            trainr1_trainr2 = compute_answers_query_2p(entity1, [rel1, rel2], easy_ent_out, easy_ent_out)
            trainr1_missingr2 = compute_answers_query_2p(entity1, [rel1, rel2], easy_ent_out, missing_ent_out)
            missingr1_trainr2 = compute_answers_query_2p(entity1, [rel1, rel2], missing_ent_out, easy_ent_out)
            missingr1_missingr2 = compute_answers_query_2p(entity1, [rel1, rel2], missing_ent_out, missing_ent_out)


            # easy answer set
            new_easy_answer_set = easy_answer_set & answer_set
            new_hard_answer_set = answer_set - new_easy_answer_set
            # answers00x = trainr1_trainr2 - all_ent_r3
            answers01x = trainr1_missingr2 - all_ent_r3
            answers10x = missingr1_trainr2 - all_ent_r3
            answers11x = missingr1_missingr2 - all_ent_r3
            reachable_pin_pos_exist = ((answers01x | answers10x) - new_easy_answer_set) & hard_answer_set
            reachable_pin_pos_only_missing = (answers11x - reachable_pin_pos_exist - new_easy_answer_set) & hard_answer_set
            n_pin_pos_only_missing += len(reachable_pin_pos_only_missing)
            n_pin_pos_exist += len(reachable_pin_pos_exist)

            #n_tot_hard_answers_pin += len(hard_answer_set)
            queries_pin, new_easy_answer_set_pin, new_hard_answer_set_pin = add_query_answ_dict(query_structure,
                                                                                                query,
                                                                                                queries_pin,
                                                                                                new_easy_answer_set_pin,
                                                                                                new_hard_answer_set_pin,
                                                                                                new_easy_answer_set,
                                                                                                new_hard_answer_set)

            queries_pin_pos_exist, filtered_pin_pos_exist, answers_pin_pos_exist = add_query_answ_dict(query_structure,
                                                                                                       query,
                                                                                                       queries_pin_pos_exist,
                                                                                                       filtered_pin_pos_exist,
                                                                                                       answers_pin_pos_exist,
                                                                                                       answer_set - reachable_pin_pos_exist,
                                                                                                       reachable_pin_pos_exist)
            queries_pin_pos_only_missing, filtered_pin_pos_only_missing, answers_pin_pos_only_missing = add_query_answ_dict(
                query_structure,
                query,
                queries_pin_pos_only_missing,
                filtered_pin_pos_only_missing,
                answers_pin_pos_only_missing,
                answer_set - reachable_pin_pos_only_missing,
                reachable_pin_pos_only_missing)
        # pni proper formulation
        if query_structure == [['e', ['r', 'r', 'n']], ['e', ['r']]]:
            n_tot_hard_answers_pni += len(hard_answer_set)
            entity1 = query[0][0]
            rel1 = query[0][1][0]
            rel2 = query[0][1][1]  # negation
            entity2 = query[1][0]
            rel3 = query[1][1][0]

            set_rel = ({rel1, rel2, rel3})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_pni:
                    rel_per_query_pni[rel] += len(hard_answer_set)
                else:
                    rel_per_query_pni[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2})
            for ent in set_ent:
                if ent in anch_per_query_pni:
                    anch_per_query_pni[ent] += len(hard_answer_set)
                else:
                    anch_per_query_pni[ent] = len(hard_answer_set)
            # easy answer set
            new_easy_answer_set = easy_answer_set & answer_set

            all_answers_2p = compute_answers_query_2p(entity1, [rel1, rel2], all_ent_out, all_ent_out)
            easy_ent_r3 = easy_ent_out[entity2][rel3]
            missing_ent_r3 = missing_ent_out[entity2][rel3]
            answersxx0 = easy_ent_r3 - all_answers_2p
            answersxx1 = missing_ent_r3 - all_answers_2p
            reachable_pni_pos_exist = (answersxx0 - new_easy_answer_set) & hard_answer_set
            reachable_pni_pos_only_missing = (answersxx1 - new_easy_answer_set) & hard_answer_set
            n_pni_pos_only_missing += len(reachable_pni_pos_only_missing)
            n_pni_pos_exist += len(reachable_pni_pos_exist)

            queries_pni_pos_exist, filtered_pni_pos_exist, answers_pni_pos_exist = add_query_answ_dict(query_structure,
                                                                                                       query,
                                                                                                       queries_pni_pos_exist,
                                                                                                       filtered_pni_pos_exist,
                                                                                                       answers_pni_pos_exist,
                                                                                                       answer_set - reachable_pni_pos_exist,
                                                                                                       reachable_pni_pos_exist)
            queries_pni_pos_only_missing, filtered_pni_pos_only_missing, answers_pni_pos_only_missing = add_query_answ_dict(
                query_structure,
                query,
                queries_pni_pos_only_missing,
                filtered_pni_pos_only_missing,
                answers_pni_pos_only_missing,
                answer_set - reachable_pni_pos_only_missing,
                reachable_pni_pos_only_missing)

        #inp
        if query_structure == [[['e', ['r']], ['e', ['r', 'n']]], ['r']]:
            entity1 = query[0][0][0]
            rel1 = query[0][0][1][0]
            entity2 = query[0][1][0]
            rel2 = query[0][1][1][0]  # negated
            rel3 = query[1][0]

            set_rel = ({rel1, rel2, rel3})
            new_set_rel = set()
            for rel in set_rel:
                if rel % 2 != 0:
                    # rel is the inverse
                    rel = rel - 1
                new_set_rel.update({rel})
            set_rel = new_set_rel
            for rel in set_rel:
                if rel in rel_per_query_inp:
                    rel_per_query_inp[rel] += len(hard_answer_set)
                else:
                    rel_per_query_inp[rel] = len(hard_answer_set)
            set_ent = ({entity1, entity2})
            for ent in set_ent:
                if ent in anch_per_query_inp:
                    anch_per_query_inp[ent] += len(hard_answer_set)
                else:
                    anch_per_query_inp[ent] = len(hard_answer_set)

            all_ent_r2 = all_ent_out[entity2][rel2]
            train_ent_r1 = easy_ent_out[entity1][rel1]
            missing_ent_r1 = missing_ent_out[entity1][rel1]

            # easy answer set
            new_easy_answer_set = easy_answer_set & answer_set
            new_hard_answer_set = answer_set - new_easy_answer_set

            answers_0x0 = set()
            answers_0x1 = set()
            answers_1x0 = set()
            answers_1x1 = set()
            answers_0x = train_ent_r1 - all_ent_r2
            for ele in answers_0x:
                answers_0x0.update(easy_ent_out[ele][rel3])
            for ele in answers_0x:
                answers_0x1.update(missing_ent_out[ele][rel3])
            answers_1x = missing_ent_r1 - all_ent_r2
            for ele in answers_1x:
                answers_1x0.update(easy_ent_out[ele][rel3])
            for ele in answers_1x:
                answers_1x1.update(missing_ent_out[ele][rel3])

            reachable_inp_pos_exist = ((answers_0x1 | answers_1x0) - new_easy_answer_set) & hard_answer_set
            reachable_inp_pos_only_missing = (answers_1x1 - reachable_inp_pos_exist - new_easy_answer_set) & hard_answer_set
            n_inp_pos_only_missing += len(reachable_inp_pos_only_missing)
            n_inp_pos_exist += len(reachable_inp_pos_exist)

            n_tot_hard_answers_inp += len(hard_answer_set)
            queries_inp, new_easy_answer_set_inp, new_hard_answer_set_inp = add_query_answ_dict(query_structure,
                                                                                                query,
                                                                                                queries_inp,
                                                                                                new_easy_answer_set_inp,
                                                                                                new_hard_answer_set_inp,
                                                                                                new_easy_answer_set,
                                                                                                new_hard_answer_set)
            queries_inp_pos_exist, filtered_inp_pos_exist, answers_inp_pos_exist = add_query_answ_dict(query_structure,
                                                                                                       query,
                                                                                                       queries_inp_pos_exist,
                                                                                                       filtered_inp_pos_exist,
                                                                                                       answers_inp_pos_exist,
                                                                                                       answer_set - reachable_inp_pos_exist,
                                                                                                       reachable_inp_pos_exist)
            queries_inp_pos_only_missing, filtered_inp_pos_only_missing, answers_inp_pos_only_missing = add_query_answ_dict(
                query_structure,
                query,
                queries_inp_pos_only_missing,
                filtered_inp_pos_only_missing,
                answers_inp_pos_only_missing,
                answer_set - reachable_inp_pos_only_missing,
                reachable_inp_pos_only_missing)
        num_sampled += 1


    if n_1p_1p != 0:
        print("----1p----")
        print("Number of answers 1p: " + str(n_1p_1p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_1p))

        ## compute top rel_per_query_1p
        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_1p, 5)
        perc_top_rel = (top_k_values[0] * 100) / n_tot_hard_answers_1p
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_1p
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_1p, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_1p
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

        #writefiles
        directory_path = filepath_query_red + "/" + "1p"
        #filenames = ["1p1p_"]
        datas = [[queries_1p_1p,answers_1p_1p_filters,answers_1p_1p]]
        subtasks = ["1p"]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_1p_2p != 0:
        print("----2p----")
        print("Number of answers 2p: " + str(n_2p_2p))
        print("Number of answers 1p: " + str(n_1p_2p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_2p))

        ## compute top rel_per_query_2p
        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_2p, 5)
        perc_top_rel = (top_k_values[0] * 100) / n_tot_hard_answers_2p
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_2p
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_2p, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_2p
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

        n_tot_reduced = n_1p_2p
        n_tot_true = n_2p_2p
        print("----cardinality reduced queries----")
        print("Number of 2p queries having: intermediate entity==0: " + str((n_2p_0c_reduced*100)/n_tot_reduced))
        print("Number of 2p queries having: intermediate entity==1: " + str((n_2p_1c_reduced*100)/n_tot_reduced))
        print("Number of 2p queries having: 1<intermediate entity<=9: " + str((n_2p_2c_reduced*100)/n_tot_reduced))
        print("Number of 2p queries having: 9<intermediate entity<=99: " + str((n_2p_10c_reduced*100)/n_tot_reduced))
        print("Number of 2p queries having: intermediate entity>=100: " + str((n_2p_100c_reduced*100)/n_tot_reduced))
        print("----cardinality true queries----")
        print("Number of 2p queries having: intermediate entity==0: " + str((n_2p_0c_true*100)/n_tot_true))
        print("Number of 2p queries having: intermediate entity==1: " + str((n_2p_1c_true*100)/n_tot_true))
        print("Number of 2p queries having: 1<intermediate entity<=9: " + str((n_2p_2c_true*100)/n_tot_true))
        print("Number of 2p queries having: 9<intermediate entity<=99: " + str((n_2p_10c_true*100)/n_tot_true))
        print("Number of 2p queries having: intermediate entity>=100: " + str((n_2p_100c_true*100)/n_tot_true))

        #writefiles
        directory_path = filepath_query_red + "/" + "2p"
        #filenames = ["2p2p_","2p1p_"]
        datas = [[queries_2p_2p,answers_2p_2p_filters,answers_2p_2p],[queries_2p_1p,answers_2p_1p_filters,answers_2p_1p]]
        subtasks = ["2p", "1p"]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        # writefiles cardinality

        directory_path = filepath_card + "/" + "2p"
        subtasks = ["0c_reduced_", "1c_reduced_","2c_reduced_","10c_reduced_", "100c_reduced_"]
        datas = [[queries_2p_0c_reduced, answers_2p_0c_filters_reduced, answers_2p_0c_reduced],
                 [queries_2p_1c_reduced, answers_2p_1c_filters_reduced, answers_2p_1c_reduced],
                 [queries_2p_2c_reduced, answers_2p_2c_filters_reduced, answers_2p_2c_reduced],
                 [queries_2p_10c_reduced, answers_2p_10c_filters_reduced, answers_2p_10c_reduced],
                 [queries_2p_100c_reduced, answers_2p_100c_filters_reduced, answers_2p_100c_reduced]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        subtasks = ["0c_true_", "1c_true_", "2c_true_", "10c_true_", "100c_true_"]
        datas = [[queries_2p_0c_true, answers_2p_0c_filters_true, answers_2p_0c_true],
                 [queries_2p_1c_true, answers_2p_1c_filters_true, answers_2p_1c_true],
                 [queries_2p_2c_true, answers_2p_2c_filters_true, answers_2p_2c_true],
                 [queries_2p_10c_true, answers_2p_10c_filters_true, answers_2p_10c_true],
                 [queries_2p_100c_true, answers_2p_100c_filters_true, answers_2p_100c_true]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        #exit()

    if n_1p_3p != 0:
        print("----3p----")
        print("Number of answers 3p: " + str(n_3p_3p))
        print("Number of answers 2p: " + str(n_2p_3p))
        print("Number of answers 1p: " + str(n_1p_3p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_3p))
        n_tot_reduced = n_1p_3p+n_2p_3p
        n_tot_true = n_3p_3p
        print("----cardinality reduced queries----")
        print("Number of 3p queries having: intermediate entity==0: " + str((n_3p_0c_reduced*100)/n_tot_reduced))
        print("Number of 3p queries having: intermediate entity==1: " + str((n_3p_1c_reduced*100)/n_tot_reduced))
        print("Number of 3p queries having: 1<intermediate entity<=9: " + str((n_3p_2c_reduced*100)/n_tot_reduced))
        print("Number of 3p queries having: 9<intermediate entity<=99: " + str((n_3p_10c_reduced*100)/n_tot_reduced))
        print("Number of 3p queries having: intermediate entity>=100: " + str((n_3p_100c_reduced*100)/n_tot_reduced))
        print("----cardinality true queries----")
        print("Number of 3p queries having: intermediate entity==0: " + str((n_3p_0c_true*100)/n_tot_true))
        print("Number of 3p queries having: intermediate entity==1: " + str((n_3p_1c_true*100)/n_tot_true))
        print("Number of 3p queries having: 1<intermediate entity<=9: " + str((n_3p_2c_true*100)/n_tot_true))
        print("Number of 3p queries having: 9<intermediate entity<=99: " + str((n_3p_10c_true*100)/n_tot_true))
        print("Number of 3p queries having: intermediate entity>=100: " + str((n_3p_100c_true*100)/n_tot_true))
        ## compute top rel_per_query_3p
        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_3p, 5)
        perc_top_rel = (top_k_values[0]*100)/n_tot_hard_answers_3p
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_3p
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_3p, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_3p
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

        # writefiles
        directory_path = filepath_query_red + "/" + "3p"
        subtasks = ["3p", "2p", "1p"]
        datas = [[queries_3p_3p, answers_3p_3p_filters, answers_3p_3p],
                 [queries_3p_2p, answers_3p_2p_filters, answers_3p_2p],
                 [queries_3p_1p, answers_3p_1p_filters, answers_3p_1p]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        directory_path = filepath_card + "/" + "3p"
        subtasks = ["ovr_c_reduced_","0c_reduced_", "1c_reduced_", "2c_reduced_", "10c_reduced_", "100c_reduced_"]
        datas = [[queries_3p_reduced, answers_3p_filters_reduced, answers_3p_reduced],
                 [queries_3p_0c_reduced, answers_3p_0c_filters_reduced, answers_3p_0c_reduced],
                 [queries_3p_1c_reduced, answers_3p_1c_filters_reduced, answers_3p_1c_reduced],
                 [queries_3p_2c_reduced, answers_3p_2c_filters_reduced, answers_3p_2c_reduced],
                 [queries_3p_10c_reduced, answers_3p_10c_filters_reduced, answers_3p_10c_reduced],
                 [queries_3p_100c_reduced, answers_3p_100c_filters_reduced, answers_3p_100c_reduced]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        subtasks = ["0c_true_", "1c_true_", "2c_true_", "10c_true_", "100c_true_"]
        datas = [[queries_3p_0c_true, answers_3p_0c_filters_true, answers_3p_0c_true],
                 [queries_3p_1c_true, answers_3p_1c_filters_true, answers_3p_1c_true],
                 [queries_3p_2c_true, answers_3p_2c_filters_true, answers_3p_2c_true],
                 [queries_3p_10c_true, answers_3p_10c_filters_true, answers_3p_10c_true],
                 [queries_3p_100c_true, answers_3p_100c_filters_true, answers_3p_100c_true]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_tot_hard_answers_4p != 0:
        print("----4p----")
        print("Number of answers 4p: " + str(n_4p_4p))
        print("Number of answers 3p: " + str(n_3p_4p))
        print("Number of answers 2p: " + str(n_2p_4p))
        print("Number of answers 1p: " + str(n_1p_4p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_4p))
        print("----cardinality reduced queries----")
        n_tot_reduced = n_1p_4p + n_2p_4p + n_3p_4p
        n_tot_true = n_4p_4p
        print("Number of 4p queries having: intermediate entity==0: " + str((n_4p_0c_reduced*100)/n_tot_reduced))
        print("Number of 4p queries having: intermediate entity==1: " + str((n_4p_1c_reduced*100)/n_tot_reduced))
        print("Number of 4p queries having: 1<intermediate entity<=9: " + str((n_4p_2c_reduced*100)/n_tot_reduced))
        print("Number of 4p queries having: 9<intermediate entity<=99: " + str((n_4p_10c_reduced*100)/n_tot_reduced))
        print("Number of 4p queries having: intermediate entity>=100: " + str((n_4p_100c_reduced*100)/n_tot_reduced))
        print("----cardinality true queries----")
        print("Number of 4p queries having: intermediate entity==0: " + str((n_4p_0c_true*100)/n_tot_true))
        print("Number of 4p queries having: intermediate entity==1: " + str((n_4p_1c_true*100)/n_tot_true))
        print("Number of 4p queries having: 1<intermediate entity<=9: " + str((n_4p_2c_true*100)/n_tot_true))
        print("Number of 4p queries having: 9<intermediate entity<=99: " + str((n_4p_10c_true*100)/n_tot_true))
        print("Number of 4p queries having: intermediate entity>=100: " + str((n_4p_100c_true*100)/n_tot_true))
        ## compute top rel_per_query_4p
        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_4p, 5)
        perc_top_rel = (top_k_values[0]*100)/n_tot_hard_answers_4p
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_4p
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_4p, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_4p
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))
        # writefiles
        directory_path = filepath_query_red + "/" + "4p"
        subtasks = ["4p", "3p", "2p", "1p"]
        datas = [[queries_4p_4p, answers_4p_4p_filters, answers_4p_4p],
                 [queries_4p_3p, answers_4p_3p_filters, answers_4p_3p],
                 [queries_4p_2p, answers_4p_2p_filters, answers_4p_2p],
                 [queries_4p_1p, answers_4p_1p_filters, answers_4p_1p]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        directory_path = filepath_card + "/" + "4p"
        subtasks = ["ovr_c_reduced_","0c_reduced_", "1c_reduced_", "2c_reduced_", "10c_reduced_", "100c_reduced_"]
        datas = [[queries_4p_reduced, answers_4p_filters_reduced, answers_4p_reduced],
                 [queries_4p_0c_reduced, answers_4p_0c_filters_reduced, answers_4p_0c_reduced],
                 [queries_4p_1c_reduced, answers_4p_1c_filters_reduced, answers_4p_1c_reduced],
                 [queries_4p_2c_reduced, answers_4p_2c_filters_reduced, answers_4p_2c_reduced],
                 [queries_4p_10c_reduced, answers_4p_10c_filters_reduced, answers_4p_10c_reduced],
                 [queries_4p_100c_reduced, answers_4p_100c_filters_reduced, answers_4p_100c_reduced]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        subtasks = ["0c_true_", "1c_true_", "2c_true_", "10c_true_", "100c_true_"]
        datas = [[queries_4p_0c_true, answers_4p_0c_filters_true, answers_4p_0c_true],
                 [queries_4p_1c_true, answers_4p_1c_filters_true, answers_4p_1c_true],
                 [queries_4p_2c_true, answers_4p_2c_filters_true, answers_4p_2c_true],
                 [queries_4p_10c_true, answers_4p_10c_filters_true, answers_4p_10c_true],
                 [queries_4p_100c_true, answers_4p_100c_filters_true, answers_4p_100c_true]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_2i_1i != 0:
        print("----2i----")
        print("Number of answers 2i: " + str(n_2i_2i))
        print("Number of answers 1p: " + str(n_2i_1i))
        print("Total number of hard answers: " + str(n_tot_hard_answers_2i))
        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_2i, 5)
        perc_top_rel = (top_k_values[0]*100)/n_tot_hard_answers_2i
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_2i
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_2i, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_2i
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))
        # writefiles
        directory_path = filepath_query_red + "/" + "2i"
        subtasks = ["2i", "1p"]
        datas = [[queries_2i_2i, answers_2i_2i_filters, answers_2i_2i],
                 [queries_2i_1p, answers_2i_1p_filters, answers_2i_1p]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_3i_1i != 0:
        print("----3i----")
        print("Number of answers 3i: " + str(n_3i_3i))
        print("Number of answers 2i: " + str(n_3i_2i))
        print("Number of answers 1p: " + str(n_3i_1i))
        print("Total number of hard answers: " + str(n_tot_hard_answers_3i))

        ## compute top rel_per_query_3i
        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_3i, 5)
        perc_top_rel = (top_k_values[0]*100)/n_tot_hard_answers_3i
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_3i
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_3i, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_3i
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))


        # writefiles
        directory_path = filepath_query_red + "/" + "3i"
        subtasks = ["3i", "2i", "1p"]
        datas = [[queries_3i_3i, answers_3i_3i_filters, answers_3i_3i],
                 [queries_3i_2i, answers_3i_2i_filters, answers_3i_2i],
                 [queries_3i_1p, answers_3i_1p_filters, answers_3i_1p]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        directory_path = filepath_card + "/" + "3i"
        subtasks = ["ovr_c_reduced_"]
        datas = [[queries_3i_reduced, answers_3i_filters_reduced, answers_3i_reduced]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_tot_hard_answers_4i != 0:
        print("----4i----")
        print("Number of answers 4i: " + str(n_4i_4i))
        print("Number of answers 3i: " + str(n_3i_4i))
        print("Number of answers 2i: " + str(n_2i_4i))
        print("Number of answers 1p: " + str(n_1p_4i))
        print("Total number of hard answers: " + str(n_tot_hard_answers_4i))

        ## compute top rel_per_query_3i
        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_4i, 5)
        perc_top_rel = (top_k_values[0]*100)/n_tot_hard_answers_4i
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_4i
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_4i, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_4i
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))


        # writefiles
        directory_path = filepath_query_red + "/" + "4i"
        subtasks = ["4i", "3i", "2i", "1p"]
        datas = [[queries_4i_4i, answers_4i_4i_filters, answers_4i_4i],
                [queries_4i_3i, answers_4i_3i_filters, answers_4i_3i],
                 [queries_4i_2i, answers_4i_2i_filters, answers_4i_2i],
                 [queries_4i_1p, answers_4i_1p_filters, answers_4i_1p]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        directory_path = filepath_card + "/" + "4i"
        subtasks = ["ovr_c_reduced_"]
        datas = [[queries_4i_reduced, answers_4i_filters_reduced, answers_4i_reduced]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_pi_1p != 0:
        print("----pi----")
        print("Number of answers pi: " + str(n_pi_pi))
        print("Number of answers 2i: " + str(n_pi_2i))
        print("Number of answers 2p: " + str(n_pi_2p))
        print("Number of answers 1p: " + str(n_pi_1p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_pi))
        n_tot_reduced = n_pi_1p + n_pi_2p + n_pi_2i
        n_tot_true = n_pi_pi
        print("----cardinality reduced queries----")
        print("Number of pi queries having: intermediate entity==0: " + str((n_pi_0c_reduced*100)/n_tot_reduced))
        print("Number of pi queries having: intermediate entity==1: " + str((n_pi_1c_reduced*100)/n_tot_reduced))
        print("Number of pi queries having: 1<intermediate entity<=9: " + str((n_pi_2c_reduced*100)/n_tot_reduced))
        print("Number of pi queries having: 9<intermediate entity<=99: " + str((n_pi_10c_reduced*100)/n_tot_reduced))
        print("Number of pi queries having: intermediate entity>=100: " + str((n_pi_100c_reduced*100)/n_tot_reduced))
        print("----cardinality true queries----")
        print("Number of pi queries having: intermediate entity==0: " + str((n_pi_0c_true*100)/n_tot_true))
        print("Number of pi queries having: intermediate entity==1: " + str((n_pi_1c_true*100)/n_tot_true))
        print("Number of pi queries having: 1<intermediate entity<=9: " + str((n_pi_2c_true*100)/n_tot_true))
        print("Number of pi queries having: 9<intermediate entity<=99: " + str((n_pi_10c_true*100)/n_tot_true))
        print("Number of pi queries having: intermediate entity>=100: " + str((n_pi_100c_true*100)/n_tot_true))

        ## compute top rel_per_query_pi
        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_pi, 5)
        perc_top_rel = (top_k_values[0]*100)/n_tot_hard_answers_pi
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_pi
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_pi, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_pi
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

        directory_path = filepath_query_red + "/" + "pi"
        subtasks = ["pi", "2p", "2i", "1p"]
        datas = [[queries_pi_pi, answers_pi_pi_filters, answers_pi_pi],
                 [queries_pi_2p, answers_pi_2p_filters, answers_pi_2p],
                 [queries_pi_2i, answers_pi_2i_filters, answers_pi_2i],
                 [queries_pi_1p, answers_pi_1p_filters, answers_pi_1p]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        directory_path = filepath_card + "/" + "pi"
        subtasks = ["ovr_c_reduced_","0c_reduced_", "1c_reduced_", "2c_reduced_", "10c_reduced_", "100c_reduced_"]
        datas = [[queries_pi_reduced, answers_pi_filters_reduced, answers_pi_reduced],
                 [queries_pi_0c_reduced, answers_pi_0c_filters_reduced, answers_pi_0c_reduced],
                 [queries_pi_1c_reduced, answers_pi_1c_filters_reduced, answers_pi_1c_reduced],
                 [queries_pi_2c_reduced, answers_pi_2c_filters_reduced, answers_pi_2c_reduced],
                 [queries_pi_10c_reduced, answers_pi_10c_filters_reduced, answers_pi_10c_reduced],
                 [queries_pi_100c_reduced, answers_pi_100c_filters_reduced, answers_pi_100c_reduced]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        subtasks = ["0c_true_", "1c_true_", "2c_true_", "10c_true_", "100c_true_"]
        datas = [[queries_pi_0c_true, answers_pi_0c_filters_true, answers_pi_0c_true],
                 [queries_pi_1c_true, answers_pi_1c_filters_true, answers_pi_1c_true],
                 [queries_pi_2c_true, answers_pi_2c_filters_true, answers_pi_2c_true],
                 [queries_pi_10c_true, answers_pi_10c_filters_true, answers_pi_10c_true],
                 [queries_pi_100c_true, answers_pi_100c_filters_true, answers_pi_100c_true]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_ip_1p != 0:
        print("----ip----")
        print("Number of answers ip: " + str(n_ip_ip))
        print("Number of answers 2i: " + str(n_ip_2i))
        print("Number of answers 2p: " + str(n_ip_2p))
        print("Number of answers 1p: " + str(n_ip_1p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_ip))
        n_tot_reduced = n_ip_1p + n_ip_2p + n_ip_2i
        n_tot_true = n_ip_ip
        print("----cardinality reduced queries----")
        print("Number of ip queries having: intermediate entity==0: " + str((n_ip_0c_reduced*100)/n_tot_reduced))
        print("Number of ip queries having: intermediate entity==1: " + str((n_ip_1c_reduced*100)/n_tot_reduced))
        print("Number of ip queries having: 1<intermediate entity<=9: " + str((n_ip_2c_reduced*100)/n_tot_reduced))
        print("Number of ip queries having: 9<intermediate entity<=99: " + str((n_ip_10c_reduced*100)/n_tot_reduced))
        print("Number of ip queries having: intermediate entity>=100: " + str((n_ip_100c_reduced*100)/n_tot_reduced))
        print("----cardinality true queries----")
        print("Number of ip queries having: intermediate entity==0: " + str((n_ip_0c_true*100)/n_tot_true))
        print("Number of ip queries having: intermediate entity==1: " + str((n_ip_1c_true*100)/n_tot_true))
        print("Number of ip queries having: 1<intermediate entity<=9: " + str((n_ip_2c_true*100)/n_tot_true))
        print("Number of ip queries having: 9<intermediate entity<=99: " + str((n_ip_10c_true*100)/n_tot_true))
        print("Number of ip queries having: intermediate entity>=100: " + str((n_ip_100c_true*100)/n_tot_true))

        ## compute top rel_per_query_ip
        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_ip, 5)
        perc_top_rel = (top_k_values[0]*100)/n_tot_hard_answers_ip
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_ip
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_ip, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_ip
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

        directory_path = filepath_query_red + "/" + "ip"
        subtasks = ["ip", "2p", "2i", "1p"]
        datas = [[queries_ip_ip, answers_ip_ip_filters, answers_ip_ip],
                 [queries_ip_2p, answers_ip_2p_filters, answers_ip_2p],
                 [queries_ip_2i, answers_ip_2i_filters, answers_ip_2i],
                 [queries_ip_1p, answers_ip_1p_filters, answers_ip_1p]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        directory_path = filepath_card + "/" + "ip"
        subtasks = ["ovr_c_reduced_","0c_reduced_", "1c_reduced_", "2c_reduced_", "10c_reduced_", "100c_reduced_"]
        datas = [[queries_ip_reduced, answers_ip_filters_reduced, answers_ip_reduced],
                 [queries_ip_0c_reduced, answers_ip_0c_filters_reduced, answers_ip_0c_reduced],
                 [queries_ip_1c_reduced, answers_ip_1c_filters_reduced, answers_ip_1c_reduced],
                 [queries_ip_2c_reduced, answers_ip_2c_filters_reduced, answers_ip_2c_reduced],
                 [queries_ip_10c_reduced, answers_ip_10c_filters_reduced, answers_ip_10c_reduced],
                 [queries_ip_100c_reduced, answers_ip_100c_filters_reduced, answers_ip_100c_reduced]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        subtasks = ["0c_true_", "1c_true_", "2c_true_", "10c_true_", "100c_true_"]
        datas = [[queries_ip_0c_true, answers_ip_0c_filters_true, answers_ip_0c_true],
                 [queries_ip_1c_true, answers_ip_1c_filters_true, answers_ip_1c_true],
                 [queries_ip_2c_true, answers_ip_2c_filters_true, answers_ip_2c_true],
                 [queries_ip_10c_true, answers_ip_10c_filters_true, answers_ip_10c_true],
                 [queries_ip_100c_true, answers_ip_100c_filters_true, answers_ip_100c_true]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_tot_hard_answers_up != 0:
        print("----up----")
        print("Number of answers up: " + str(n_up_up))
        print("Number of answers 2u: " + str(n_up_2u))
        print("Number of answers with a random link in the union: " + str(n_up_2p))
        print("Number of answers 1p: " + str(n_up_1p))
        print("Number of answers 0p: " + str(n_up_0p))
        print("Total number of hard answers: " + str(n_tot_hard_answers_up))
        n_tot_reduced = n_up_1p+n_up_2u
        n_tot_true = n_up_up
        print("----cardinality reduced queries----")
        print("Number of up queries having: intermediate entity==0: " + str((n_up_0c_reduced*100)/n_tot_reduced))
        print("Number of up queries having: intermediate entity==1: " + str((n_up_1c_reduced*100)/n_tot_reduced))
        print("Number of up queries having: 1<intermediate entity<=9: " + str((n_up_2c_reduced*100)/n_tot_reduced))
        print("Number of up queries having: 9<intermediate entity<=99: " + str((n_up_10c_reduced*100)/n_tot_reduced))
        print("Number of up queries having: intermediate entity>=100: " + str((n_up_100c_reduced*100)/n_tot_reduced))
        print("----cardinality true queries----")
        print("Number of up queries having: intermediate entity==0: " + str((n_up_0c_true*100)/n_tot_true))
        print("Number of up queries having: intermediate entity==1: " + str((n_up_1c_true*100)/n_tot_true))
        print("Number of up queries having: 1<intermediate entity<=9: " + str((n_up_2c_true*100)/n_tot_true))
        print("Number of up queries having: 9<intermediate entity<=99: " + str((n_up_10c_true*100)/n_tot_true))
        print("Number of up queries having: intermediate entity>=100: " + str((n_up_100c_true*100)/n_tot_true))

        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_up, 5)
        perc_top_rel = (top_k_values[0] * 100) / n_tot_hard_answers_up
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_up
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_up, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_up
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))


        directory_path = filepath_query_red + "/" + "up"
        subtasks = ["ov", "up", "2u", "1p"]
        datas = [[queries_up_overall, answers_up_filters_overall, answers_up_overall],
                 [queries_up_up, answers_up_up_filters, answers_up_up],
                 [queries_up_2u, answers_up_2u_filters, answers_up_2u],
                 [queries_up_1p, answers_up_1p_filters, answers_up_1p]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        directory_path = filepath_card + "/" + "up"
        subtasks = ["ovr_c_reduced_","0c_reduced_", "1c_reduced_", "2c_reduced_", "10c_reduced_", "100c_reduced_"]
        datas = [[queries_up_reduced, answers_up_filters_reduced, answers_up_reduced],
                 [queries_up_0c_reduced, answers_up_0c_filters_reduced, answers_up_0c_reduced],
                 [queries_up_1c_reduced, answers_up_1c_filters_reduced, answers_up_1c_reduced],
                 [queries_up_2c_reduced, answers_up_2c_filters_reduced, answers_up_2c_reduced],
                 [queries_up_10c_reduced, answers_up_10c_filters_reduced, answers_up_10c_reduced],
                 [queries_up_100c_reduced, answers_up_100c_filters_reduced, answers_up_100c_reduced]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
        subtasks = ["0c_true_", "1c_true_", "2c_true_", "10c_true_", "100c_true_"]
        datas = [[queries_up_0c_true, answers_up_0c_filters_true, answers_up_0c_true],
                 [queries_up_1c_true, answers_up_1c_filters_true, answers_up_1c_true],
                 [queries_up_2c_true, answers_up_2c_filters_true, answers_up_2c_true],
                 [queries_up_10c_true, answers_up_10c_filters_true, answers_up_10c_true],
                 [queries_up_100c_true, answers_up_100c_filters_true, answers_up_100c_true]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])

    if n_tot_hard_answers_2u >0:
        print("----2u----")
        print("Number of answers 2u: " + str(n_2u_2u))
        print("Total number of hard answers: " + str(n_tot_hard_answers_2u))

        directory_path = filepath_query_red + "/" + "2u"
        subtasks = ["2u"]
        datas = [[queries_2u_2u, answers_2u_2u_filters, answers_2u_2u]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])

        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_2u, 5)
        perc_top_rel = (top_k_values[0] * 100) / n_tot_hard_answers_2u
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_2u
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_2u, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_2u
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

    if n_tot_hard_answers_2in >0:
        print("----2in----")
        print("Tot hard answers:"+str(n_tot_hard_answers_2in))
        print("----positive reasoning tree----")
        print("Number of answers reachable with at least an existing link in the positive part:" + str(n_2in_pos_exist) + "  %"+str((n_2in_pos_exist/n_tot_hard_answers_2in)*100))
        print("Number of answers reachable with only missing links in the positive part:" + str(n_2in_pos_only_missing) + "  %"+str((n_2in_pos_only_missing/n_tot_hard_answers_2in)*100))

        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_2in, 5)
        perc_top_rel = (top_k_values[0] * 100) / n_tot_hard_answers_2in
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_2in
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_2in, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_2in
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

        directory_path = filepath_query_red + "/" + "2in"
        subtasks = ["fixed"]
        datas = [[queries_2in, new_easy_answer_set_2in, new_hard_answer_set_2in]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_tot_hard_answers_3in >0:
        print("----3in----")
        print("Tot hard answers:" + str(n_tot_hard_answers_3in))
        print("----positive reasoning tree----")
        print("Number of answers reachable with at least an existing link in the positive part:" + str(
            n_3in_pos_exist) + "  %" + str((n_3in_pos_exist / n_tot_hard_answers_3in) * 100))
        print("Number of answers reachable with only missing links in the positive part:" + str(
            n_3in_pos_only_missing) + "  %" + str((n_3in_pos_only_missing / n_tot_hard_answers_3in) * 100))

        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_3in, 5)
        perc_top_rel = (top_k_values[0] * 100) / n_tot_hard_answers_3in
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_3in
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_3in, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_3in
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

        # writefiles
        directory_path = filepath_query_red + "/" + "3in"
        subtasks = ["fixed", "pos-exist", "pos-only-miss"]
        datas = [[queries_3in, new_easy_answer_set_3in, new_hard_answer_set_3in],
                 [queries_3in_pos_exist, filtered_3in_pos_exist, answers_3in_pos_exist],
                 [queries_3in_pos_only_missing, filtered_3in_pos_only_missing, answers_3in_pos_only_missing]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_tot_hard_answers_pin >0:
        print("----pin----")
        print("Tot hard answers:" + str(n_tot_hard_answers_pin))
        print("----positive reasoning tree----")
        print("Number of answers reachable with at least an existing link in the positive part:" + str(
            n_pin_pos_exist) + "  %" + str((n_pin_pos_exist / n_tot_hard_answers_pin) * 100))
        print("Number of answers reachable with only missing links in the positive part:" + str(
            n_pin_pos_only_missing) + "  %" + str((n_pin_pos_only_missing / n_tot_hard_answers_pin) * 100))

        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_pin, 5)
        perc_top_rel = (top_k_values[0] * 100) / n_tot_hard_answers_pin
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_pin
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_pin, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_pin
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

        # writefiles
        directory_path = filepath_query_red + "/" + "pin"
        subtasks = ["fixed", "pos-exist", "pos-only-miss"]
        datas = [[queries_pin, new_easy_answer_set_pin, new_hard_answer_set_pin],
                 [queries_pin_pos_exist, filtered_pin_pos_exist, answers_pin_pos_exist],
                 [queries_pin_pos_only_missing, filtered_pin_pos_only_missing, answers_pin_pos_only_missing]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_tot_hard_answers_pni >0:
        print("----pni----")
        print("Tot hard answers:"+str(n_tot_hard_answers_pni))
        print("----positive reasoning tree----")
        print("Number of answers reachable with at least an existing link in the positive part:" + str(
            n_pni_pos_exist) + "  %" + str((n_pni_pos_exist / n_tot_hard_answers_pni) * 100))
        print("Number of answers reachable with only missing links in the positive part:" + str(
            n_pni_pos_only_missing) + "  %" + str((n_pni_pos_only_missing / n_tot_hard_answers_pni) * 100))

        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_pni, 5)
        perc_top_rel = (top_k_values[0] * 100) / n_tot_hard_answers_pni
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_pni
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_pni, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_pni
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

        # writefiles
        directory_path = filepath_query_red + "/" + "pni"
        subtasks = ["pos-only-miss"]
        datas = [[queries_pni_pos_only_missing, filtered_pni_pos_only_missing, answers_pni_pos_only_missing]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
    if n_tot_hard_answers_inp >0:
        print("----inp----")
        print("Tot hard answers:"+str(n_tot_hard_answers_inp))
        print("----positive reasoning tree----")
        print("Number of answers reachable with at least an existing link in the positive part:" + str(
            n_inp_pos_exist) + "  %" + str((n_inp_pos_exist / n_tot_hard_answers_inp) * 100))
        print("Number of answers reachable with only missing links in the positive part:" + str(
            n_inp_pos_only_missing) + "  %" + str((n_inp_pos_only_missing / n_tot_hard_answers_inp) * 100))

        top_k_keys, top_k_values = top_k_dict_values_sorting(rel_per_query_inp, 5)
        perc_top_rel = (top_k_values[0] * 100) / n_tot_hard_answers_inp
        top_rel_name = top_k_keys[0]
        id2rel = "./data/{}/{}".format(dataset, "id2rel.pkl")
        with open(id2rel, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present relation name: " + str(data_dict[top_rel_name]))
        print("Percentage of the relation name wrt total number of (q,a) pair: " + str(perc_top_rel))

        ## compute top anch_per_query_inp
        top_k_anch_keys, top_k_anch_values = top_k_dict_values_sorting(anch_per_query_inp, 5)
        perc_top_ent = (top_k_anch_values[0] * 100) / n_tot_hard_answers_inp
        top_ent_name = top_k_anch_keys[0]
        id2ent = "./data/{}/{}".format(dataset, "id2ent.pkl")
        with open(id2ent, 'rb') as f:
            data_dict = pickle.load(f)
        print("Most present anchor: " + str(data_dict[top_ent_name]))
        print("Percentage of the anchor wrt total number of (q,a) pair: " + str(perc_top_ent))

        # writefiles
        directory_path = filepath_query_red + "/" + "inp"
        subtasks = ["fixed", "pos-exist", "pos-only-miss"]
        datas = [[queries_inp, new_easy_answer_set_inp, new_hard_answer_set_inp],
                 [queries_inp_pos_exist, filtered_inp_pos_exist, answers_inp_pos_exist],
                 [queries_inp_pos_only_missing, filtered_inp_pos_only_missing, answers_inp_pos_only_missing]]
        for i in range(len(subtasks)):
            sub_task_directory_path = directory_path + "/" + subtasks[i]
            save_pkl_file(sub_task_directory_path, filenames, datas[i])
def top_k_dict_values_sorting(d, k):
    # Sort the dictionary by values in descending order and get the top-k items
    sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=True)[:k]
    # Extract the keys and values from the sorted items
    top_k_keys = [item[0] for item in sorted_items]
    top_k_values = [item[1] for item in sorted_items]
    return top_k_keys, top_k_values
def divby1(x): return x
def read_queries(dataset, query_structures, gen_num, max_ans_num, gen_train, gen_valid, gen_test, query_names,
                 save_name):
    base_path = './data/%s' % dataset
    indexified_files = ['train.txt', 'valid.txt', 'test.txt']
    if gen_train or gen_valid:
        train_ent_in, easy_ent_out = construct_graph(base_path, indexified_files[:1])  # ent_in
    if gen_valid or gen_test:
        valid_ent_in, valid_ent_out = construct_graph(base_path, indexified_files[:2])
        valid_only_ent_in, valid_only_ent_out = construct_graph(base_path, indexified_files[1:2])

    if gen_test:
        test_ent_in, test_ent_out = construct_graph(base_path, indexified_files[:3])
        test_only_ent_in, missing_ent_out = construct_graph(base_path, indexified_files[2:3])

    idx = 0
    query_name = query_names[idx] if save_name else str(idx)

    name_to_save = query_name
    set_logger("./data/{}/".format(dataset), name_to_save)

    if gen_train:
        train_queries_path = "./data/{}/train-queries.pkl".format(dataset)
        train_queries = read_pkl_file(train_queries_path)
        for query_structure in query_structures:
            queries = train_queries[list2tuple(query_structure)]
            find_answers(query_structure, queries, train_ent_in, easy_ent_out, train_ent_in, easy_ent_out,train_ent_in, easy_ent_out, 'train', dataset)

    if gen_valid:
        valid_queries_path = "./data/{}/valid-queries.pkl".format(dataset)
        valid_queries = read_pkl_file(valid_queries_path)
        #valid_answ_path = "./data/{}/valid-hard-answers.pkl".format(dataset)
        #hard_answer_set = read_pkl_file(valid_answ_path)
        for query_structure in query_structures:
            queries = valid_queries[list2tuple(query_structure)]
            find_answers(query_structure, queries, valid_only_ent_in, valid_only_ent_out, valid_ent_in, valid_ent_out,
                         train_ent_in, easy_ent_out, 'valid', dataset)

    if gen_test:
        test_queries_path = "./data/{}/test-queries.pkl".format(dataset)
        test_queries = read_pkl_file(test_queries_path)
        test_answ_path = "./data/{}/test-hard-answers.pkl".format(dataset)
        computed_hard_answer_set = read_pkl_file(test_answ_path) # only a subset of the answers of each query is considered; we retrieve the considered answers
        for query_structure in query_structures:
            queries = test_queries[list2tuple(query_structure)]
            find_answers(query_structure, queries,computed_hard_answer_set,
                         test_only_ent_in,
                         missing_ent_out,
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
@click.option('--gen_train_num', default=10000)
@click.option('--gen_valid_num', default=10000)
@click.option('--gen_test_num', default=10000)
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
    if index_only:
        index_dataset(dataset, reindex)
        exit(-1)

    e = 'e'
    r = 'r'
    n = 'n'
    u = 'u'

    query_structures = [
        #[e, [r]],
        [e, [r, r]],

      [e, [r, r, r]],
        ##[e, [r, r, r, r]],
        #[[e, [r]], [e, [r]]],
        #[[e, [r]], [e, [r]], [e, [r]]],
        #[[e, [r]], [e, [r]], [e, [r]], [e, [r]]],
        [[e, [r, r]], [e, [r]]],
        [[[e, [r]], [e, [r]]], [r]],
        # negation
        #[[e, [r]], [e, [r, n]]],
        #[[e, [r]], [e, [r]], [e, [r, n]]],
        #[[e, [r, r]], [e, [r, n]]],
        #[[e, [r, r, n]], [e, [r]]],
        #[[[e, [r]], [e, [r, n]]], [r]],
        # union
        #[[e, [r]], [e, [r]], [u]],
        [[[e, [r]], [e, [r]], [u]], [r]]
    ]

    query_names = ['2p', '3p', '2i', '3i','pi', 'ip', '3in', 'pin', 'inp', '2u', 'up']
    #query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip','2u', 'up']
    #query_names = ['2i']
    # generate_queries(dataset, query_structures, [gen_train_num, gen_valid_num, gen_test_num], max_ans_num, gen_train, gen_valid, gen_test, query_names[gen_id:gen_id+1], save_name)
    read_queries(dataset, query_structures, [gen_train_num, gen_valid_num, gen_test_num],
                 max_ans_num, gen_train, gen_valid, gen_test, query_names[gen_id:gen_id + 1], save_name)


if __name__ == '__main__':
    main()
