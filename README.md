# is-cqa-complex
Is Complex Query Answering Complex?

# KGReasoning

This repo contains several algorithms for multi-hop reasoning on knowledge graphs, including the official PyTorch implementation of [Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs](https://arxiv.org/abs/2010.11465) and a PyTorch implementation of [Complex Query Answering with Neural Link Predictors](https://arxiv.org/abs/2011.03459).

**Models**
- [x] [CQD](https://arxiv.org/abs/2011.03459)
- [x] [BetaE](https://arxiv.org/abs/2010.11465)
- [x] [Query2box](https://arxiv.org/abs/2002.05969)
- [x] [GQE](https://arxiv.org/abs/1806.01445)

**KG Data**

The KG data (FB15k, FB15k-237, NELL995) mentioned in the BetaE paper and the Query2box paper can be downloaded [here](http://snap.stanford.edu/betae/KG_data.zip). Note the two use the same training queries, but the difference is that the valid/test queries in BetaE paper have a maximum number of answers, making it more realistic.

Each folder in the data represents a KG, including the following files.
- `train.txt/valid.txt/test.txt`: KG edges
- `id2rel/rel2id/ent2id/id2ent.pkl`: KG entity relation dicts
- `train-queries/valid-queries/test-queries.pkl`: `defaultdict(set)`, each key represents a query structure, and the value represents the instantiated queries
- `train-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`)
- `valid-easy-answers/test-easy-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`) / valid graph (edges in `train.txt`+`valid.txt`)
- `valid-hard-answers/test-hard-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the **additional** answers obtained in the validation graph (edges in `train.txt`+`valid.txt`) / test graph (edges in `train.txt`+`valid.txt`+`test.txt`)

We represent the query structures using a tuple in case we run out of names :), (credits to @michiyasunaga). For example, 1p queries: (e, (r,)) and 2i queries: ((e, (r,)),(e, (r,))). Check the code for more details.


** How to run **
1. Download data --> download.sh
2. Run read_queries_pair.py for the datasets FB15k-237-betae, FB15k-betae, NELL-betae. The script will add the folder "test-query-reduction" in each folder of the datasets. 
The statistics of the reductions are in the folder "reduction statistics".
3. Train the model you want to test. For CQD, you can use the provided pre-trained models (more details in CQD.md).
4. To test the subset of 2p queries that can be reduced to 1p using CQD it is sufficient to set --tasks -2p and --subtask 1p. If subtask is None, then the whole orginal set of queries will be tested
statistics.txt
python3 main.py --do_test --data_path data/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd 
--tasks 2p --subtask -1p --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-betae --cqd discrete --cqd-t-norm prod --cqd-k 1024



**Citations**

If you use this repo, please cite the following paper.

```
@inproceedings{
 ren2020beta,
 title={Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs},
 author={Hongyu Ren and Jure Leskovec},
 booktitle={Neural Information Processing Systems},
 year={2020}
}
```


