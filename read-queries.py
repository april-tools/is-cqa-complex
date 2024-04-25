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

    log_file = os.path.join(save_path, '%s.log'%(query_name))

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
        print ("index file exists")
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
                print ('[%d/%d]'%(i, file_len), end='\r')
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

    print ('num entity: %d, num relation: %d'%(len(ent2id), len(rel2id)))
    print ("indexing finished!!")

def construct_graph(base_path, indexified_files):
    #knowledge graph
    #kb[e][rel] = set([e, e, e])
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
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

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

    with open('./data/%s/%s-queries.pkl'%(dataset, name), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-tp-answers.pkl'%(dataset, name), 'wb') as f:
        pickle.dump(tp_answers, f)
    with open('./data/%s/%s-fn-answers.pkl'%(dataset, name), 'wb') as f:
        pickle.dump(fn_answers, f)
    with open('./data/%s/%s-fp-answers.pkl'%(dataset, name), 'wb') as f:
        pickle.dump(fp_answers, f)
    print (num_more_answer)


def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl_file(dataset,filepath,name_to_save,data):
    with open(filepath % (dataset, name_to_save), 'wb') as f:
        pickle.dump(data, f)
    return data

def find_answers(query_structure, queries, missing_ent_in, missing_ent_out, all_ent_in, all_ent_out, easy_ent_in, easy_ent_out, mode):
    '''
    missing_ent = entities related only to the validation/test set
    all_ent = entities related to the train + validation/test set
    easy_ent = entities related only to train

    '''
    random.seed(0)
    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
    easy_ans_num, hard_ans_num, very_hard_ans_num, answer_set_q1_num = [], [], [], []
    easy_answers = defaultdict(set)
    very_hard_answers = defaultdict(set)
    not_so_hard_answers = defaultdict(set)
    queries_very_hard_answers = set()
    queries_not_so_hard_answers = set()
    easy_answ_for_not_so_hard_answers = defaultdict(set)
    all_answers_set = defaultdict(set)
    #fp_answers = defaultdict(set)
    hard_answers = defaultdict(set)
    easy_answ_for_very_hard_answers = defaultdict(set)
    # I can substitute this while with the reading of the existing valid/test queries
    noccA = 0
    noccB = 0
    noccC = 0
    noccD =0
    noccE = 0

    for query in queries:
        query = tuple2list(query)
        answer_set = achieve_answer(query, all_ent_in, all_ent_out)
        easy_answer_set = achieve_answer(query, easy_ent_in, easy_ent_out)
        very_hard_answer_set = achieve_answer(query, missing_ent_in, missing_ent_out)


        if query_structure == ['e', ['r', 'r']]:
            entity = query[0]
            rel1 = query[1][0]
            rel2 = query[1][1]
            # count A
            if len(easy_answer_set)==0 and len(answer_set-very_hard_answer_set) ==0 and len(very_hard_answer_set) !=0:
                answer_set_q2_1 = set()
                answer_set_q1_1 = easy_ent_out[entity][rel1]
                for ent_int in answer_set_q1_1:
                    answer_set_q2_1.update(missing_ent_out[ent_int][rel2])

                answer_set_q2_2 = set()
                answer_set_q1_2 = missing_ent_out[entity][rel1]
                for ent_int in answer_set_q1_2:
                    answer_set_q2_2.update(easy_ent_out[ent_int][rel2])

                if len(answer_set_q2_1) == 0 and len(answer_set_q2_2) == 0:
                    queries_very_hard_answers.add(list2tuple(query))
                    very_hard_answers[list2tuple(query)] = very_hard_answer_set - easy_answer_set
                    noccA+=1
            #count B
            if len(easy_answer_set) == 0 and len(answer_set - very_hard_answer_set) == 0 and len(very_hard_answer_set) != 0:
                answer_set_q2_1 = set()
                answer_set_q1_1 = easy_ent_out[entity][rel1]
                for ent_int in answer_set_q1_1:
                    answer_set_q2_1.update(missing_ent_out[ent_int][rel2])

                answer_set_q2_2 = set()
                answer_set_q1_2 = missing_ent_out[entity][rel1]
                for ent_int in answer_set_q1_2:
                    answer_set_q2_2.update(easy_ent_out[ent_int][rel2])

                if len(answer_set_q2_1) > 0 or len(answer_set_q2_2) > 0:
                    queries_very_hard_answers.add(list2tuple(query))
                    very_hard_answers[list2tuple(query)] = very_hard_answer_set - easy_answer_set
                    noccB += 1

            #count C
            if len(easy_answer_set) == 0 and len(very_hard_answer_set) == 0:
                noccC+=1


            #count D
            if len(easy_answer_set) != 0:
                noccD+=1

            #count E
            if len(easy_answer_set) == 0 and len(answer_set-very_hard_answer_set) !=0 and len(very_hard_answer_set) !=0:
                noccE+=1
        elif query_structure == [['e', ['r']], ['e', ['r']]]:
            # count A
            if len(easy_answer_set) == 0 and len(answer_set - very_hard_answer_set) == 0 and len(
                    very_hard_answer_set) != 0:
                    noccA += 1
            

            # count C
            if len(easy_answer_set) == 0 and len(very_hard_answer_set) == 0:
                noccC += 1

            # count D
            if len(easy_answer_set) != 0:
                noccD += 1

            # count E
            if len(easy_answer_set) == 0 and len(answer_set - very_hard_answer_set) != 0 and len(
                    very_hard_answer_set) != 0:
                noccE += 1
        num_sampled += 1
        #answer_set_q1_num.append(len(answer_set_q1))
        easy_ans_num.append(len(easy_answers[list2tuple(query)]))
        hard_ans_num.append(len(hard_answers[list2tuple(query)]))

    print("Number of queries of type A: " + str(noccA))
    print("Number of queries of type B: " + str(noccB))
    print("Number of queries of type C: " + str(noccC))
    print("Number of queries of type D: " + str(noccD))
    print("Number of queries of type E: " + str(noccE))

    if mode == "train":
        return None,None,None,None,None,None,None,all_answers_set
    else:
        return queries_very_hard_answers, queries_not_so_hard_answers, hard_answers, easy_answers, very_hard_answers, easy_answ_for_very_hard_answers, not_so_hard_answers, easy_answ_for_not_so_hard_answers



def read_queries(dataset, query_structures, gen_num, max_ans_num, gen_train, gen_valid, gen_test, query_names, save_name):
    base_path = './data/%s'%dataset
    indexified_files = ['train.txt', 'valid.txt', 'test.txt']
    if gen_train or gen_valid:
        train_ent_in, train_ent_out = construct_graph(base_path, indexified_files[:1]) # ent_in
    if gen_valid or gen_test:
        valid_ent_in, valid_ent_out = construct_graph(base_path, indexified_files[:2])
        valid_only_ent_in, valid_only_ent_out = construct_graph(base_path, indexified_files[1:2])

    if gen_test:
        test_ent_in, test_ent_out = construct_graph(base_path, indexified_files[:3])
        test_only_ent_in, test_only_ent_out = construct_graph(base_path, indexified_files[2:3])

    idx =0
    query_name = query_names[idx] if save_name else str(idx)

    name_to_save = query_name
    set_logger("./data/{}/".format(dataset), name_to_save)

    if gen_valid:
        valid_queries_path = "./data/{}/valid-queries.pkl".format(dataset)
        valid_queries = read_pkl_file(valid_queries_path)
        for query_structure in query_structures:
            queries = valid_queries[list2tuple(query_structure)]
            find_answers(query_structure, queries, valid_only_ent_in, valid_only_ent_out,valid_ent_in, valid_ent_out,train_ent_in, train_ent_out,'valid')

    if gen_test:
        test_queries_path = "./data/{}/test-queries.pkl".format(dataset)
        test_queries = read_pkl_file(test_queries_path)
        for query_structure in query_structures:
            queries = test_queries[list2tuple(query_structure)]
            find_answers(query_structure, queries,test_only_ent_in,test_only_ent_out,test_ent_in, test_ent_out,valid_ent_in, valid_ent_out,'test')
            print ('%s read queries with structure %s'%(gen_num, query_structure))


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
def main(dataset, seed, gen_train_num, gen_valid_num, gen_test_num, max_ans_num, reindex, gen_train, gen_valid, gen_test, gen_id, save_name, index_only):
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
    # only 2p and 2i implemented
    query_structures = [
                        #[e, [r]],
                        [e, [r, r]],
                        #[e, [r, r, r]],
                        [[e, [r]], [e, [r]]],
                        #[[e, [r]], [e, [r]], [e, [r]]],
                        #[[e, [r, r]], [e, [r]]],
                        #[[[e, [r]], [e, [r]]], [r]],
                        # negation
                        #[[e, [r]], [e, [r, n]]],
                        #[[e, [r]], [e, [r]], [e, [r, n]]],
                        #[[e, [r, r]], [e, [r, n]]],
                        #[[e, [r, r, n]], [e, [r]]],
                        #[[[e, [r]], [e, [r, n]]], [r]],
                        # union
                        #[[e, [r]], [e, [r]], [u]],
                        #[[[e, [r]], [e, [r]], [u]], [r]]
                    ]


    query_names = ['2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']

    read_queries(dataset, query_structures, [gen_train_num, gen_valid_num, gen_test_num],
                     max_ans_num, gen_train, gen_valid, gen_test, query_names[gen_id:gen_id + 1], save_name)

if __name__ == '__main__':
    main()