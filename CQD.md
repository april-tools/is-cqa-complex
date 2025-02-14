This repository is based on the official implementation of CQD, [**Complex Query Answering with Neural Link Predictors**](https://openreview.net/forum?id=Mos9F9kDwkz).

```bibtex
@inproceedings{
    arakelyan2021complex,
    title={Complex Query Answering with Neural Link Predictors},
    author={Erik Arakelyan and Daniel Daza and Pasquale Minervini and Michael Cochez},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=Mos9F9kDwkz}
}
```

In this work we present a variation of CQD, namely CQD-Hybrid, which reuses a pretrained link predictor to answer complex queries, by scoring atom predicates independently and aggregating the scores via t-norms and t-conorms, and additionally gives maximum score to the existing triples find in each query atom. 
We also added the code (in discrete.py and base.py) for answering 4p and 4i queries, namely a 4-path long sequence and an intersection of four links in the target variable.


We use the same pre-trained models available made available by the authors for CQD:

## 1. Download the pre-trained models

Download the models available in the release https://github.com/april-tools/is-cqa-complex/releases/tag/models-v1.0 

## Evaluate the model
In job-NELL995.sh, you find an example script used for evaluating CQD and CQD hybrid on the existing benchmarks, including their query reductions (as explained in the paper).
By setting the hyperparameter "cqd_max_norm=1'', you will launch the original CQD, while by setting, "cqd_max_norm=0.9" you will automatically use CQD-Hybrid, which will assign a score=1 to the existing triples.
For the full list of hyperparameters for the old and new benchmarks please refer to the Appendix F of the paper.
