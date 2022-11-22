
## Extrinsic Evaluation on Recommendation

## Datasets

We release the aligned knowledge graph with different plausiblity and typicality filters in the following [shared folders](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jbai_connect_ust_hk/EqqnjNof5B1Ot682jWI-XWgBEpXiroLgRs9BwRUTcXyDug?e=KL4ESr). Please download all the pickle and gpickle files and put in the same directory. 


## Implementation

### 1. Preprocessing the graph and textual features from knowledge graph

```
python preprocess.py
```

This python script will generate six json files containing the graph and textual features for the eventualities and entities. Then we use the following scripts to generate the modified transE features to capture both structural and textual features. 

```
bash run_modified_transE.sh

```

### 2. Train the recommendation models

For the NCF and WnD baselines, you can run the following scripts


```
bash run_electronic_ncf.sh
bash run_cloth_ncf.sh
bash run_electronic_wnd.sh
bash run_cloth_wnd.sh

```

For the text-only and graph-only models, please use the following scripts

```
bash run_electronic_wnd_graph_only.sh
bash run_cloth_wnd_graph_only.sh
bash run_electronic_wnd_text_only.sh
bash run_cloth_wnd_text_only.sh

```


For the knowledge graph features using different thresholds use the following scripts

```
bash run_electronic_wnd_graph_00.sh
bash run_electronic_wnd_graph_05.sh
bash run_electronic_wnd_graph_05_05.sh
bash run_electronic_wnd_graph_09.sh
bash run_electronic_wnd_graph_09_09.sh

bash run_cloth_wnd_graph_00.sh
bash run_cloth_wnd_graph_05.sh
bash run_cloth_wnd_graph_05_05.sh
bash run_cloth_wnd_graph_09.sh
bash run_cloth_wnd_graph_09_09.sh

```

### Troubleshooting

If you have any comments or questions regarding to the recommendation part, please contact Jiaxin Bai via (jbai@connect.ust.hk). 
