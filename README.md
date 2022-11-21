# FolkScope

Sourcecode and datasets for the paper "FolkScope: Intention Knowledge Graph Construction for Discovering E-commerce Commonsense" ([arXiv](https://arxiv.org/pdf/2211.08316.pdf))

## Datasets

We release the annotated training datasets and the whole poplulated generations with both plausibility and typicality scores in the [shared folders](https://hkustconnect-my.sharepoint.com/personal/cyuaq_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcyuaq%5Fconnect%5Fust%5Fhk%2FDocuments%2Ffolkscope&ga=1).


## Implementation

### 1. Prompting Generation

```bash
bash scripts/run_generation.sh
```

### 2. Classifier Training and Inference
```
bash scripts/run_training.sh
bash scripts/run_inference.sh
```

## Citation

Please cite the following paper if you found our method helpful. Thanks !

```
@inproceedings{yu2022folkscope,
  title={FolkScope: Intention Knowledge Graph Construction for Discovering E-commerce Commonsense},
  author={Changlong Yu and Weiqi Wang and Xin Liu and Jiaxin Bai and Yangqiu Song and Zheng Li and Yifan Gao and Tianyu Cao and Bing Yin},
  year={2022}
}
```

