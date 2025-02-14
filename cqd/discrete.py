# -*- coding: utf-8 -*-
import copy
import datetime
import pickle
import random

import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
import numpy as np
from typing import Callable, Tuple, Optional

def score_candidates(queries: Tensor,
                     filters,
                     s_emb: Tensor,
                     p_emb: Tensor,
                     candidates_emb: Tensor,
                     k: Optional[int],
                     max_norm: Optional[int],
                     max_k: Optional[int],
                     entity_embeddings: nn.Module,
                     scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tuple[
    Tensor, Optional[Tensor], Optional[Tensor]]:
    batch_size = max(s_emb.shape[0], p_emb.shape[0])
    embedding_size = s_emb.shape[1]
    def reshape(emb: Tensor) -> Tensor:
        if emb.shape[0] < batch_size:
            n_copies = batch_size // emb.shape[0]
            emb = emb.reshape(-1, 1, embedding_size).repeat(1, n_copies, 1).reshape(-1, embedding_size)
        return emb

    s_emb = reshape(s_emb)
    p_emb = reshape(p_emb)
    nb_entities = candidates_emb.shape[0]

    x_k_emb_3d = None
    atom_k_indices = None
    # [B, N]
    atom_scores_2d = scoring_function(s_emb, p_emb, candidates_emb)
    if max_norm!=1: #implies we're using CQD-hybrid
        n_existing_per_queries = list(np.zeros(len(queries), dtype=int))
        for i, query in enumerate(queries):
            if (query[0].item(), query[1].item()) in filters.keys():
                existing_candidates = filters[(query[0].item(), query[1].item())]
                atom_scores_2d[i, existing_candidates] = 1.0  # maximum_score, scores are always normalized between 0 and max_score in [0,0.9]
                n_existing_per_queries[i] = len(existing_candidates)
        #print(n_existing_per_queries)
        atom_k_scores_2d = atom_scores_2d
        #print(queries)
        if k is not None:
            if len(n_existing_per_queries) > 0 and max_k is not None:
                max_k_per_queries = np.max(np.array(n_existing_per_queries))
                k_ = min(max_k_per_queries + k, max_k, nb_entities) #max_k is the upper bound, as pointed out in Appendix C
            elif len(n_existing_per_queries) > 0 and max_k is None:
                max_k_per_queries = np.max(np.array(n_existing_per_queries))
                k_ = min(max_k_per_queries + k, nb_entities)
            elif len(n_existing_per_queries) == 0 and max_k is None:
                k_ = min(k, nb_entities)
            elif len(n_existing_per_queries) == 0 and max_k is not None:
                k_ = min(k, nb_entities,max_k)
            # [B, K], [B, K]

            if len(queries) > 1: 
                atom_k_scores_2d_ = torch.zeros(atom_scores_2d.size()[0], k_)
                atom_k_indices_2d_ = torch.zeros(atom_scores_2d.size()[0], k_, dtype=torch.int32)
                for idx, n_per_query in enumerate(n_existing_per_queries):
                    k__ = min(k + n_existing_per_queries[idx], nb_entities, max_k)  # if no existing n_existing_per_queries[idx] =0!
                    topkscores, topkindices= torch.topk(atom_scores_2d[idx].unsqueeze(0), k=k__, dim=1)
                    atom_k_scores_2d_[idx,:k__] = topkscores
                    atom_k_indices_2d_[idx,:k__] = topkindices
                atom_k_scores_2d = atom_k_scores_2d_
                atom_k_indices = atom_k_indices_2d_
            else:
                atom_k_scores_2d, atom_k_indices = torch.topk(atom_scores_2d, k=k_, dim=1)
            x_k_emb_3d = entity_embeddings(atom_k_indices)
    else:
        atom_k_scores_2d = atom_scores_2d

        if k is not None:
            k_ = min(k, nb_entities)
            # [B, K], [B, K]
            atom_k_scores_2d, atom_k_indices = torch.topk(atom_scores_2d, k=k_, dim=1)
            # [B, K, E]
            x_k_emb_3d = entity_embeddings(atom_k_indices)


    return atom_k_scores_2d, x_k_emb_3d, atom_k_indices




def query_1p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             max_k: int,
             max_norm: int,
             filters,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tensor:
    s_emb = entity_embeddings(queries[:, 0])
    p_emb = predicate_embeddings(queries[:, 1])
    candidates_emb = entity_embeddings.weight

    assert queries.shape[1] == 2

    res, _, _ = score_candidates(queries=queries, filters=filters, s_emb=s_emb, p_emb=p_emb,
                                 candidates_emb=candidates_emb, k=None,
                                 entity_embeddings=entity_embeddings,
                                 scoring_function=scoring_function,max_norm=max_norm, max_k = max_k)
    return res


def query_2p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             filters,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             max_k: int,
             max_norm: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])
    query1 = queries[:, :2]

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, K], [B, K, E]
    atom1_k_scores_2d, x1_k_emb_3d, int_entities = score_candidates(queries=query1, filters=filters, s_emb=s_emb,
                                                                    p_emb=p1_emb,
                                                                    candidates_emb=candidates_emb, k=k,
                                                                    entity_embeddings=entity_embeddings,
                                                                    scoring_function=scoring_function,max_norm=max_norm, max_k=max_k)

    # [B * K, E]
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)

    # [B * K, N]
    queries2 = []
    for atom_k_index in int_entities[0]:
        query2 = [atom_k_index.item(), queries[:, 2].item()]
        queries2.append(query2)
    queries2 = torch.tensor(queries2)
    # [B * K, N]
    atom2_scores_2d, _, _ = score_candidates(queries=queries2, filters=filters, s_emb=x1_k_emb_2d, p_emb=p2_emb,
                                             candidates_emb=candidates_emb, k=None,
                                             entity_embeddings=entity_embeddings,
                                             scoring_function=scoring_function,max_norm=max_norm, max_k = max_k)

    # [B, K] -> [B, K, N]
    atom1_scores_3d = atom1_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)
    # [B * K, N] -> [B, K, N]
    atom2_scores_3d = atom2_scores_2d.reshape(batch_size, -1, nb_entities)

    res = t_norm(atom1_scores_3d, atom2_scores_3d)

    # [B, K, N] -> [B, N]
    res, _ = torch.max(res, dim=1)
    return res


#query_3p
def query_3p(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 filters,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 k: int,
                 max_norm: int,
                 max_k: int,
                 t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])
    p3_emb = predicate_embeddings(queries[:, 3])
    query1 = queries[:, :2]
    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]
    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]
    # [B, K], [B, K, E]
    atom1_k_scores_2d, x1_k_emb_3d, int_entities1 = score_candidates(queries=query1, filters=filters, s_emb=s_emb,
                                                                     p_emb=p1_emb,
                                                                     candidates_emb=candidates_emb, k=k,
                                                                     entity_embeddings=entity_embeddings,
                                                                     scoring_function=scoring_function,max_norm=max_norm, max_k = max_k)

    # [B * K, E]
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)
    # [B * K, K], [B * K, K, E]
    queries2 = []
    for atom_k_index in int_entities1.view(-1):
        query2 = [atom_k_index.item(), queries[:, 2].item()]
        queries2.append(query2)
    queries2 = torch.tensor(queries2)
    atom2_k_scores_2d, x2_k_emb_3d, int_entities2 = score_candidates(queries=queries2, filters=filters,
                                                                     s_emb=x1_k_emb_2d, p_emb=p2_emb,
                                                                     candidates_emb=candidates_emb, k=k,
                                                                     entity_embeddings=entity_embeddings,
                                                                     scoring_function=scoring_function, max_norm=max_norm, max_k = max_k)
    # [B * K * K, E]
    x2_k_emb_2d = x2_k_emb_3d.reshape(-1, emb_size)
    # [B * K * K, N]
    queries3 = []
    for atom_k_index in int_entities2.view(-1):
        query3 = [atom_k_index.item(), queries[:, 3].item()]
        queries3.append(query3)
    queries3 = torch.tensor(queries3)

    atom3_scores_2d, _, _ = score_candidates(queries=queries3, filters=filters,
                                             s_emb=x2_k_emb_2d, p_emb=p3_emb,
                                             candidates_emb=candidates_emb, k=None,
                                             entity_embeddings=entity_embeddings,
                                             scoring_function=scoring_function,max_norm=max_norm, max_k = max_k)
    # [B, K] -> [B, K, N]
    atom1_scores_3d = atom1_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)

    # [B * K * K, N] -> [B, K * K, N]
    atom3_scores_3d = atom3_scores_2d.reshape(batch_size, -1, nb_entities)
    atom2_scores_3d = atom2_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)
    atom1_scores_3d = atom1_scores_3d.repeat(1, atom3_scores_3d.shape[1] // atom1_scores_3d.shape[1], 1)

    res = t_norm(atom1_scores_3d, atom2_scores_3d)
    res = t_norm(res, atom3_scores_3d)

    # [B, K, N] -> [B, N]
    res, _ = torch.max(res, dim=1)

    return res


def query_4p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             filters,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             max_norm: int,
             max_k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])
    p3_emb = predicate_embeddings(queries[:, 3])
    p4_emb = predicate_embeddings(queries[:, 3])
    query1 = queries[:, :2]
    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]
    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]
    # [B, K], [B, K, E]
    atom1_k_scores_2d, x1_k_emb_3d, int_entities1 = score_candidates(queries=query1, filters=filters, s_emb=s_emb,
                                                                     p_emb=p1_emb,
                                                                     candidates_emb=candidates_emb, k=k,
                                                                     entity_embeddings=entity_embeddings,
                                                                     scoring_function=scoring_function,
                                                                     max_norm=max_norm, max_k=max_k)

    # [B * K, E]
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)
    # [B * K, K], [B * K, K, E]
    queries2 = []
    for atom_k_index in int_entities1.view(-1):
        query2 = [atom_k_index.item(), queries[:, 2].item()]
        queries2.append(query2)
    queries2 = torch.tensor(queries2)
    atom2_k_scores_2d, x2_k_emb_3d, int_entities2 = score_candidates(queries=queries2, filters=filters,
                                                                     s_emb=x1_k_emb_2d, p_emb=p2_emb,
                                                                     candidates_emb=candidates_emb, k=k,
                                                                     entity_embeddings=entity_embeddings,
                                                                     scoring_function=scoring_function,
                                                                     max_norm=max_norm, max_k=max_k)

    # [B * K * K, E]
    x2_k_emb_2d = x2_k_emb_3d.reshape(-1, emb_size)
    # [B * K * K, N]
    queries3 = []
    for atom_k_index in int_entities2.view(-1):
        query3 = [atom_k_index.item(), queries[:, 3].item()]
        queries3.append(query3)
    queries3 = torch.tensor(queries3)

    atom3_k_scores_2d, x3_k_emb_3d, int_entities3 = score_candidates(queries=queries3, filters=filters,
                                                                     s_emb=x2_k_emb_2d, p_emb=p3_emb,
                                                                     candidates_emb=candidates_emb, k=k,
                                                                     entity_embeddings=entity_embeddings,
                                                                     scoring_function=scoring_function,
                                                                     max_norm=max_norm, max_k=max_k)

    # [B * K * K, E]
    x3_k_emb_2d = x3_k_emb_3d.reshape(-1, emb_size)
    # [B * K * K, N]
    queries4 = []
    for atom_k_index in int_entities3.view(-1):
        query4 = [atom_k_index.item(), queries[:, 4].item()]
        queries4.append(query4)
    queries4 = torch.tensor(queries4)

    atom4_scores_2d, _, _ = score_candidates(queries=queries4, filters=filters,
                                             s_emb=x3_k_emb_2d, p_emb=p4_emb,
                                             candidates_emb=candidates_emb, k=None,
                                             entity_embeddings=entity_embeddings,
                                             scoring_function=scoring_function, max_norm=max_norm, max_k=max_k)

    # [B, K] -> [B, K, N]
    atom1_scores_3d = atom1_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)
    # [B * K * K, N] -> [B, K * K, N]
    atom2_scores_3d = atom2_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)
    atom3_scores_3d = atom3_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)

    atom4_scores_3d = atom4_scores_2d.reshape(batch_size, -1, nb_entities)
    atom1_scores_3d = atom1_scores_3d.repeat(1, atom3_scores_3d.shape[1] // atom1_scores_3d.shape[1], 1)
    atom2_scores_3d = atom1_scores_3d.repeat(1, atom4_scores_3d.shape[1] // atom1_scores_3d.shape[1], 1)

    res = t_norm(atom1_scores_3d, atom2_scores_3d)
    res = t_norm(res, atom3_scores_3d)
    res = t_norm(atom3_scores_3d, atom4_scores_3d)

    # [B, K, N] -> [B, N]
    res, _ = torch.max(res, dim=1)

    return res
def query_2i(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             filters,
             max_k: int,
             max_norm: int,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    scores_1 = query_1p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function, max_k=max_k, max_norm=max_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function, max_k=max_k,max_norm=max_norm)

    res = t_norm(scores_1, scores_2)

    return res


def query_3i(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             filters,
             max_k: int,
             max_norm: int,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    scores_1 = query_1p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function, max_k=max_k,max_norm=max_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function, max_k=max_k,max_norm=max_norm)
    scores_3 = query_1p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 4:6], scoring_function=scoring_function, max_k=max_k,max_norm=max_norm)

    res = t_norm(scores_1, scores_2)
    res = t_norm(res, scores_3)

    return res

def query_4i(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             filters,
             max_k: int,
             max_norm: int,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    scores_1 = query_1p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function, max_k=max_k,max_norm=max_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function, max_k=max_k,max_norm=max_norm)
    scores_3 = query_1p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 4:6], scoring_function=scoring_function, max_k=max_k,max_norm=max_norm)
    scores_4 = query_1p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 6:8], scoring_function=scoring_function, max_k=max_k,max_norm=max_norm)
    res = t_norm(scores_1, scores_2)
    res = t_norm(res, scores_3)
    res = t_norm(res, scores_4)
    return res

def query_ip(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             filters,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             max_k: int,
             max_norm: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    # [B, N]
    scores_1 = query_2i(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:4], scoring_function=scoring_function, t_norm=t_norm, max_k=max_k,max_norm=max_norm)

    # [B, E]
    p_emb = predicate_embeddings(queries[:, 4])

    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]
    num_existing_answers = 0
    if max_norm!=1: #if cqd hybrid, count existing intermediate answers
        occurrences_mask = torch.eq(scores_1, 1.0)
        num_existing_answers = torch.sum(occurrences_mask).item()

    if num_existing_answers > 0 and max_k is not None:
        k_ = min(num_existing_answers + k, nb_entities, max_k)
    elif num_existing_answers > 0 and max_k is None:
        k_ = min(num_existing_answers + k, nb_entities)
    elif num_existing_answers == 0 and max_k is None:
        k_ = min(k, nb_entities)
    elif num_existing_answers == 0 and max_k is not None:
        k_ = min(k, nb_entities, max_k)


    #k_ = min(num_existing_answers + k, max_k, nb_entities)  # I have to add all the answers having scores = 1

    # [B, K], [B, K]
    scores_1_k, scores_1_k_indices = torch.topk(scores_1, k=k_, dim=1)

    # [B, K, E]
    scores_1_k_emb = entity_embeddings(scores_1_k_indices)

    # [B * K, E]
    scores_1_k_emb_2d = scores_1_k_emb.reshape(batch_size * k_, emb_size)

    # [B * K, N]
    query_p = []
    for atom_k_index in scores_1_k_indices.view(-1):
        query_ = [atom_k_index.item(), queries[:, 4].item()]
        query_p.append(query_)
    query_p = torch.tensor(query_p)
    scores_2, _, _ = score_candidates(s_emb=scores_1_k_emb_2d, queries=query_p, p_emb=p_emb, filters=filters,
                                      candidates_emb=e_emb, k=None,
                                      entity_embeddings=entity_embeddings, scoring_function=scoring_function,max_norm=max_norm, max_k = max_k)

    # [B * K, N]
    scores_1_k = scores_1_k.reshape(batch_size, k_, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, k_, nb_entities)

    res = t_norm(scores_1_k, scores_2)
    res, _ = torch.max(res, dim=1)

    return res


def query_pi(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             filters,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             max_k: int,
             max_norm: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    scores_1 = query_2p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:3], scoring_function=scoring_function, k=k, t_norm=t_norm, max_k = max_k,max_norm=max_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, filters=filters, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 3:5], scoring_function=scoring_function, max_k = max_k,max_norm=max_norm)

    res = t_norm(scores_1, scores_2)

    return res


def query_2u_dnf(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 filters,
                 max_k: int,
                 max_norm: int,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 t_conorm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], filters=filters, scoring_function=scoring_function, max_k = max_k,max_norm=max_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], filters=filters, scoring_function=scoring_function, max_k = max_k,max_norm=max_norm)

    res = t_conorm(scores_1, scores_2)

    return res


def query_up_dnf(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 filters,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 k: int,
                 max_k: int,
                 max_norm: int,
                 t_norm: Callable[[Tensor, Tensor], Tensor],
                 t_conorm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    # [B, N]
    scores_1 = query_2u_dnf(entity_embeddings=entity_embeddings, filters=filters,
                            predicate_embeddings=predicate_embeddings,
                            queries=queries[:, 0:4], scoring_function=scoring_function, t_conorm=t_conorm, max_k = max_k,max_norm=max_norm)

    # [B, E]
    p_emb = predicate_embeddings(queries[:, 5])

    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]
    num_existing_answers = 0
    if max_norm!=1: #if cqd hybrid, count existing intermediate answers
        occurrences_mask = torch.eq(scores_1, 1.0)
        num_existing_answers = torch.sum(occurrences_mask).item()

    if num_existing_answers > 0 and max_k is not None:
        k_ = min(num_existing_answers + k, nb_entities, max_k)
    elif num_existing_answers > 0 and max_k is None:
        k_ = min(num_existing_answers + k, nb_entities)
    elif num_existing_answers == 0 and max_k is None:
        k_ = min(k, nb_entities)
    elif num_existing_answers == 0 and max_k is not None:
        k_ = min(k, nb_entities, max_k)

    # [B, K], [B, K]
    scores_1_k, scores_1_k_indices = torch.topk(scores_1, k=k_, dim=1)
    # [B, K, E]
    scores_1_k_emb = entity_embeddings(scores_1_k_indices)

    # [B * K, E]
    scores_1_k_emb_2d = scores_1_k_emb.reshape(batch_size * k_, emb_size)
    query_i = []
    for atom_k_index in scores_1_k_indices.view(-1):
        query_ = [atom_k_index.item(), queries[:, 5].item()]
        query_i.append(query_)
    query_i = torch.tensor(query_i)
    # [B * K, N]
    scores_2, _, _ = score_candidates(s_emb=scores_1_k_emb_2d, p_emb=p_emb, queries=query_i, filters=filters,
                                      candidates_emb=e_emb, k=None,
                                      entity_embeddings=entity_embeddings, scoring_function=scoring_function, max_k = max_k,max_norm=max_norm)

    # [B * K, N]
    scores_1_k = scores_1_k.reshape(batch_size, k_, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, k_, nb_entities)

    res = t_norm(scores_1_k, scores_2)
    res, _ = torch.max(res, dim=1)

    return res