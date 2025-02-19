# is-cqa-complex
Is Complex Query Answering Complex? https://arxiv.org/pdf/2410.12537 

## KGReasoning

This repo contains several algorithms for multi-hop reasoning on knowledge graphs, including the official PyTorch implementation of [Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs](https://arxiv.org/abs/2010.11465) and a PyTorch implementation of [Complex Query Answering with Neural Link Predictors](https://arxiv.org/abs/2011.03459).

### Models
- [x] [CQD](https://arxiv.org/abs/2011.03459)
- [x] [BetaE](https://arxiv.org/abs/2010.11465)
- [x] [Query2box](https://arxiv.org/abs/2002.05969)
- [x] [GQE](https://arxiv.org/abs/1806.01445)

### Analysis on benchmarks
To reproduce our stratified analysis on both new and old benchmarks, or to execute it for other benchmarks, run `read_queries_pair.py`. 
To change the benchmark it is sufficient to set the `--dataset` parameter. 
The script will generate the files to perform both the stratified analysis and the analysis on cardinality, which you can find in the folder `benchmark/test-query-red` and `benchmark/test-query-card`.
We included such files for each benchmark we considered in the release https://github.com/april-tools/is-cqa-complex/releases/tag/benchs-1.0 

### New benchmarks
To generate new benchmarks following the strategy we described in the paper, run `create_queries.py`, which is a modified version of the one included in the official PyTorch implementation of [Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs].
The KG data and the benchmarks we used in this paper (FB15k237+H, NELL995+H, ICEWS18+H) can be downloaded from https://github.com/april-tools/is-cqa-complex/releases/tag/benchs-1.0 
The folder contains both the old and the new benchmarks, including the benchmark files for their stratified analysis.



### Pre-trained models
All pre-trained models we used in this paper can be downloaded from https://github.com/april-tools/is-cqa-complex/releases/tag/models-v1.0 


### Running CQD and CQD-Hybrid
1. Download new benchmarks and pre-trained models--> see above
2. To test the subset of 2p queries that can be reduced to 1p using CQD or CQD-Hybrid it is sufficient to set `--tasks -2p` and `--subtask 1p`. For details about CQD-Hybrid see `cqd/CQD.md`
If subtask `--subtask None`, then the whole orginal set of queries will be tested, while if `--subtask New`, the new set of queries will be tested. An example for NELL995 is provided in `job-NELL995.sh`
3. A file `results.csv` containing the MRR,H@1,H@3,H@10 for every task/subtask will be created while running the script.

### Results on the new benchmarks
There is no clear state-of-the-art (SoTA) method for the new benchmarks. As shown in the table below, the Mean Reciprocal Rank (MRR) on the new benchmarks is significantly lower than the old ones. For example, for 3i queries on **FB15k-237+H**, QTO achieves an MRR of **10.1**, whereas for **FB15k237**, it was **54.6**.

---

#### **FB15k-237+H**

| Model       | 1p  | 2p  | 3p  | 2i  | 3i  | 1p2i | 2i1p | 2u  | 2u1p | 4p  | 4i  |
|------------|-----|-----|-----|-----|-----|------|------|-----|------|-----|-----|
| GNN-QE      | 42.8 | **5.2** | **4.0** | 6.0 | 8.8 | 5.6  | 9.9  | 32.5 | 10.0 | **4.3** | 20.0 |
| ULTRAQ      | 40.6 | 4.5 | 3.5 | 5.2 | 7.2 | 5.3  | 10.1 | 29.4 | 8.3  | 3.8  | 16.4 |
| CQD         | **46.7** | 4.4 | 2.4 | **11.3** | **12.8** | 6.0  | 11.5 | 40.1 | 10.6 | 1.1  | **23.8** |
| CQD-Hybrid  | **46.7** | 4.8 | 3.1 | 6.0 | 8.6 | 5.5  | 12.9 | **42.2** | **12.0** | 2.4  | 17.4 |
| ConE        | 41.8 | 4.6 | 3.9 | 9.1 | 10.3 | 3.8  | 7.9  | 22.8 | 6.0  | 3.5  | 20.3 |
| QTO         | **46.7** | 4.9 | 3.7 | 8.7 | 10.1 | **6.1**  | **13.5** | 30.6 | 11.2 | 3.9  | 20.2 |

---

#### **NELL995+H**

| Model       | 1p  | 2p  | 3p  | 2i  | 3i  | 1p2i | 2i1p | 2u  | 2u1p | 4p  | 4i  |
|------------|-----|-----|-----|-----|-----|------|------|-----|------|-----|-----|
| GNN-QE      | 53.6 | 8.0 | 6.0 | 10.7 | 13.3 | 16.0 | 13.5 | 47.5 | 9.8  | 4.7  | 19.4 |
| ULTRAQ      | 38.9 | 6.1 | 4.1 | 7.9 | 10.2 | 15.8 | 9.3  | 28.1 | 9.5  | 4.2  | 15.6 |
| CQD         | **60.4** | 9.6 | 4.2 | **18.5** | **19.6** | 18.9 | **22.6** | 46.3 | 18.5 | 2.0  | 24.8 |
| CQD-Hybrid  | **60.4** | 9.0 | 6.1 | 12.1 | 14.4 | 17.4 | 21.2 | 46.4 | **19.3** | 3.5  | 20.4 |
| ConE        | 53.1 | 7.9 | 6.7 | **21.8** | **23.6** | 14.9 | 11.8 | 39.9 | 8.8  | 5.2  | **27.6** |
| QTO         | 60.3 | **9.8** | **8.0** | 14.6 | 15.8 | **17.6** | 21.1 | **49.1** | 18.9 | **7.0**  | 20.9 |

---

#### **ICEWS18+H**

| Model       | 1p  | 2p  | 3p  | 2i  | 3i  | 1p2i | 2i1p | 2u  | 2u1p | 4p  | 4i  |
|------------|-----|-----|-----|-----|-----|------|------|-----|------|-----|-----|
| GNN-QE      | 12.2 | 0.9 | 0.5 | **16.1** | **26.5** | **19.1** | 3.5  | **7.6**  | 1.1  | 0.4  | **34.0** |
| ULTRAQ      | 6.3  | 1.2 | 1.2 | 7.0  | 11.7 | 8.8  | 1.3  | 3.3  | 1.2  | 0.8  | 15.9 |
| CQD         | **16.6** | 2.5 | **1.5** | 13.0 | 19.5 | 17.1 | **6.7** | 6.8  | **5.9**  | **1.1**  | 24.0 |
| CQD-Hybrid  | **16.6** | **2.6** | **1.5** | 15.0 | 25.5 | 17.5 | 5.8  | 6.8  | 5.6  | 0.9  | 33.2 |
| ConE        | 3.5  | 0.9 | 0.9 | 1.2  | 0.5  | 1.2  | 1.6  | 1.1  | 0.9  | 0.6  | 0.3  |
| QTO         | **16.6** | **2.6** | 1.4 | 15.7 | 25.0 | 18.4 | 6.2  | 6.7  | 4.9  | **1.1**  | 31.5 |

---







