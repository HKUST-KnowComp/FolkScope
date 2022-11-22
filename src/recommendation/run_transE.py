import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import json

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim
import random
import networkx as nx

import argparse


class TransE(nn.Module):

    def __init__(self, entity_count, relation_count, device, norm=1, dim=384, margin=1.0):
        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.dim = dim
        self.entities_emb = self._init_enitity_emb()
        self.relations_emb = self._init_relation_emb()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    def _init_enitity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.dim,
                                    padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count)
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relations_emb

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor, positive_embeddings, negative_embeddings):
        """Return model losses based on the input.
        :param positive_triplets: triplets of positive ids in Bx3 shape (B - batch, 3 - head1, head2 and relation)
        :param negative_triplets: triplets of negative ids in Bx3 shape (B - batch, 3 - head1, head2 and relation)
        
        :param positive_embeddings:  positives semantic features in Bx384 shape (B - batch, 384 - SBERT features)
        :param negative_embeddings:  negatives semantic features in Bx384 shape (B - batch, 384 - SBERT features)
        
        
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        # -1 to avoid nan for OOV vector
        self.entities_emb.weight.data[:-1, :].div_(self.entities_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

        assert positive_triplets.size()[1] == 3
        positive_distances = self._distance(positive_triplets, positive_embeddings)

        assert negative_triplets.size()[1] == 3
        negative_distances = self._distance(negative_triplets, negative_embeddings)

        return self.loss(positive_distances, negative_distances), positive_distances, negative_distances

    def predict(self, triplets: torch.LongTensor):
        """Calculated dissimilarity score for given triplets.
        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(triplets)

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, triplets, embeddings):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        assert triplets.size()[1] == 3
        heads_1 = triplets[:, 0]
        heads_2 = triplets[:, 1]
        relations = triplets[:, 2]
        tails = embeddings
       
        self.entities_emb(heads_1)
        self.entities_emb(heads_2)
        self.relations_emb(relations)
        
        return ((self.entities_emb(heads_1) + self.entities_emb(heads_2)) / 2 + self.relations_emb(relations) - tails).norm(p=self.norm,
                                                                                                          dim=1)

class TrainDataset(Dataset):
    def __init__(self, kg, eventuality_dict):
        
        
#         all_asins = set([u for u, v, a in kg.edges(data=True)])
        
        all_asins = []
        for u, v, a in kg.edges(data=True):
            asin_1, asin_2 = u.split("-")
            all_asins.append(asin_1)
            all_asins.append(asin_2)
        
        
        self.asin2id = {}
        self.all_asins = list(set(all_asins))
        
        for n in self.all_asins:
            self.asin2id[n] = len(self.asin2id)
            
        all_relations = []
        for u, v, a in kg.edges(data=True):
            for rel, value in a.items():
                all_relations.append(rel)
        self.all_relations = list(set(all_relations))
        
        self.relation2id = {}
        for n in self.all_relations:
            self.relation2id[n] = len(self.relation2id)
        
        
        self.triples = []
        for u, v, a in tqdm(kg.edges(data=True)):
            for rel, value in a.items():
                asin_1, asin_2 = u.split("-")
                self.asin2id[asin_1]
                self.asin2id[asin_2]
                self.relation2id[rel]
                eventuality_dict[v]
                self.triples.append([self.asin2id[asin_1], self.asin2id[asin_2], self.relation2id[rel], eventuality_dict[v]])
        
        self.len = len(self.triples)
            
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.triples[idx], len(self.asin2id)
    
    @staticmethod
    def collate_fn(data):
        positive_sample = [_[0][:-1] for _ in data]
        random_entity_ids = np.random.randint(data[0][1], size=len(data))
        
        _r = random.random()
        if _r < 0.50:
            negative_sample = [ [random_entity_ids[l_id], l[0][1], l[0][2]]  for l_id, l in enumerate(data)]
        else:
            random_entity_ids_2 = np.random.randint(data[0][1], size=len(data))
            negative_sample = [ [random_entity_ids[l_id], random_entity_ids_2[l_id], l[0][2]]  for l_id, l in enumerate(data)]

        return positive_sample, negative_sample, [l[0][3]  for l_id, l in enumerate(data)], [l[0][3]  for l_id, l in enumerate(data)]

class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

           
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The training transE')

  
    parser.add_argument('--total_steps', default=10000, type=int, help='train log every xx steps')
    parser.add_argument('-b', '--batch_size', default=4096, type=int)
    
    
    parser.add_argument('-d', '--domain', type=str, required=True)
    parser.add_argument('-f', '--filtered', action='store_true')
    parser.add_argument('-r', '--ratio',type=str, required=True)

    

    args = parser.parse_args()
    
    print("start training")
    
    # domain = "Electronics"
    domain = args.domain
    ratio = args.ratio
    
   
    graph = nx.read_gpickle(domain + "_" + ratio + "__filtered_kg.gpickle")

        
        
    with open(domain + "_eventuality_SBERT_feature.json", "r") as fin:
        eventuality_features = json.load(fin)

    # batch_size = 1024
    batch_size = args.batch_size

    dataset = TrainDataset(graph, eventuality_features)

    # print(dataset.asin2id)

    train_iterator = SingledirectionalOneShotIterator(DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=TrainDataset.collate_fn
            ))


    model = TransE(len(set([u for u in graph.nodes()])), len(dataset.all_relations), torch.device('cuda'))
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    STEP_LIMIT = 20000


    for step in tqdm(range(STEP_LIMIT)):

        positive_sample, negative_sample, positive_tail_embedding, negative_tail_embedding= next(train_iterator)

        positive_sample = torch.tensor(positive_sample).cuda()
        negative_sample = torch.tensor(negative_sample).cuda()

        positive_tail_embedding = torch.tensor(positive_tail_embedding).cuda()
        negative_tail_embedding = torch.tensor(negative_tail_embedding).cuda()



        loss, positive_distance, negative_distance = model(positive_sample, negative_sample, positive_tail_embedding, negative_tail_embedding)
        loss = loss.mean()

        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(loss)
            
            
    feature_dict = {}

    for asin, eid in dataset.asin2id.items():
        feature_dict[asin] = model.entities_emb.weight[eid].detach().cpu().numpy().tolist()

    
    with open(domain + "_" + ratio + "_filtered_transE_feature.json", "w") as fout:
        json.dump(feature_dict, fout)
    

    