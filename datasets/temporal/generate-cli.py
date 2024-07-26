#!/usr/bin/env python3

import os
import csv

import argparse

from typing import Tuple, List, Optional, Dict, Union

Symbol = Union[int, str]
Quad = Tuple[Symbol, Symbol, Symbol, int]


def parse_mapping(path: str) -> Dict[int, str]:
    res: Dict[int, str] = {}
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            symbol = row[0]
            idx = int(row[1])
            res[idx] = symbol
    return res

def split_by_time(quad_lst: List[Quad]) -> Tuple[List[Quad], List[Quad], List[Quad]]:
    # Sort quads by timestep 't'
    sorted_quads = sorted(quad_lst, key=lambda x: x[3])
    
    total_quads = len(sorted_quads)
    train_end = int(0.8 * total_quads)
    dev_end = train_end + int(0.1 * total_quads)
    
    train_set = sorted_quads[:train_end]
    dev_set = sorted_quads[train_end:dev_end]
    test_set = sorted_quads[dev_end:]
    
    return train_set, dev_set, test_set

def filter_minimum_timestep(quad_lst: List[Quad]) -> List[Quad]:
    min_timestep_dict = {}
    for quad in quad_lst:
        spo = (quad[0], quad[1], quad[2])
        if spo not in min_timestep_dict or quad[3] < min_timestep_dict[spo][3]:
            min_timestep_dict[spo] = quad
    return list(min_timestep_dict.values())


def parse_tsv(path: str,
              entity2idx_path: Optional[str] = None,
              relation2idx_path: Optional[str] = None) -> List[Quad]:
    
    entity2idx = relation2idx = None
    if entity2idx_path is not None and relation2idx_path is not None:
        entity2idx = parse_mapping(entity2idx_path)
        relation2idx = parse_mapping(relation2idx_path)

    res: List[Quad] = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            subject_ = int(row[0]) if entity2idx is None else entity2idx[int(row[0])]
            predicate_ = int(row[1]) if relation2idx is None else relation2idx[int(row[1])]
            object_ = int(row[2]) if entity2idx is None else entity2idx[int(row[2])]
            timestep_ = int(row[3])
            res += [(subject_, predicate_, object_, timestep_)]
    return res

def save_to_tsv(tuple_lst: List[Quad], path: str):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for tuple_ in tuple_lst:
            writer.writerow(tuple_)

def main():
    parser = argparse.ArgumentParser(description="Dataset creator")
    parser.add_argument("prefix")
    args = parser.parse_args()

    prefix = args.prefix
    path_lst = ['train.txt', 'valid.txt', 'test.txt']

    quad_lst: List[Quad] = []
    for path in path_lst:
        full_path = os.path.join(prefix, path)
        entity2idx_path = os.path.join(prefix, 'entity2id.txt')
        relation2idx_path = os.path.join(prefix, 'relation2id.txt')

        quad_lst += parse_tsv(full_path,
                              entity2idx_path=entity2idx_path,
                              relation2idx_path=relation2idx_path)
    
    print(f'{prefix} -- Number of quads before timestep filtering', len(quad_lst))
    quad_lst = filter_minimum_timestep(quad_lst=quad_lst)
    print(f'{prefix} -- Number of quads after timestep filtering', len(quad_lst))

    save_to_tsv(quad_lst=quad_lst, path=os.path.join('generated', f'{prefix}.tsv'))
    
    train_lst, dev_lst, test_lst = split_by_time(quad_lst=quad_lst)
    
    save_to_tsv(quad_lst=train_lst, path=os.path.join('generated', f'{prefix}_train.tsv'))
    save_to_tsv(quad_lst=dev_lst, path=os.path.join('generated', f'{prefix}_dev.tsv'))
    save_to_tsv(quad_lst=test_lst, path=os.path.join('generated', f'{prefix}_test.tsv'))

if __name__ == "__main__":
    main()