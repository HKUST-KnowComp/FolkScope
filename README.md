# FolkScope

Sourcecode and datasets for the paper "FolkScope: Intention Knowledge Graph Construction for E-commerce Commonsense Discovery" ([[arXiv](https://arxiv.org/pdf/2211.08316.pdf)] [[Amazon Science](https://www.amazon.science/publications/folkscope-intention-knowledge-graph-construction-for-e-commerce-commonsense-discovery)])

### News

Folkscope's extension work, [COSMO](https://dl.acm.org/doi/10.1145/3626246.3653398) has been published in the SIGMOD 2024 and we scale up the behavior type, product categories, and data annotation. [[Amazon Science Blog](https://www.amazon.science/blog/building-commonsense-knowledge-graphs-to-aid-product-recommendation)]

![Overview](figure/folkscope.png)

## Datasets

We release product metadata, the annotated training datasets and the whole poplulated generations with both plausibility and typicality scores, and recommendation data in the [shared folders](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cyuaq_connect_ust_hk/EhLWuDJtP5pPgPH27i5Oq1oBxfc0wDIqFxpvJhdPcdt9hA?e=6JROlg).


## Implementation

### Package Dependencies

* nltk
* wandb
* pandas
* sklearn
* evalaute
* datasets
* tqdm
* sentencepiece
* accelerate==0.9.0
* torch==1.10.1+cu111
* transformers==4.20.0
* python-igraph == 0.9.11
* stanfordnlp==0.2.0


### 1. Prompting Generation

```bash
bash scripts/run_generation.sh
```

### 2. Classifier Training and Inference
```bash
bash scripts/run_training.sh
bash scripts/run_inference.sh
```

### 3. Knowledge Graph Construction
Kind reminder: please ensure that you have more than 100GB memory for pattern mining. Otherwise, please set a smaller `num_workers`
```bash
bash scripts/run_mining.sh
bash scripts/run_match.sh
bash scripts/run_conceptualization.sh
```

## Citation

Please kindly cite the following paper if you found our method and resources helpful!

```
@inproceedings{yu-etal-2023-folkscope,
    title = "{F}olk{S}cope: Intention Knowledge Graph Construction for {E}-commerce Commonsense Discovery",
    author = "Yu, Changlong  and
      Wang, Weiqi  and
      Liu, Xin  and
      Bai, Jiaxin  and
      Song, Yangqiu  and
      Li, Zheng  and
      Gao, Yifan  and
      Cao, Tianyu  and
      Yin, Bing",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.76",
    pages = "1173--1191",
}
```

```
@inproceedings{yu2024cosmo,
    author = {Yu, Changlong and Liu, Xin and Maia, Jefferson and Li, Yang and Cao, Tianyu and Gao, Yifan and Song, Yangqiu and Goutam, Rahul and Zhang, Haiyang and Yin, Bing and Li, Zheng},
    title = {COSMO: A Large-Scale E-commerce Common Sense Knowledge Generation and Serving System at Amazon},
    year = {2024},
    isbn = {9798400704222},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3626246.3653398},
    doi = {10.1145/3626246.3653398},
    booktitle = {Companion of the 2024 International Conference on Management of Data},
    pages = {148–160},
    numpages = {13},
    location = {Santiago AA, Chile},
    series = {SIGMOD/PODS '24}
}
```
