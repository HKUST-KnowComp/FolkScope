
from sentence_transformers import SentenceTransformer
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

import pickle



class TransE(nn.Module):

    def __init__(self, entity_count, relation_count, device, norm=1, dim=100, margin=1.0):
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

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        """Return model losses based on the input.
        :param positive_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param negative_triplets: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        # -1 to avoid nan for OOV vector
        self.entities_emb.weight.data[:-1, :].div_(self.entities_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

        assert positive_triplets.size()[1] == 3
        positive_distances = self._distance(positive_triplets)

        assert negative_triplets.size()[1] == 3
        negative_distances = self._distance(negative_triplets)

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

    def _distance(self, triplets):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
       
        self.entities_emb(heads)
        self.relations_emb(relations)
        self.entities_emb(tails)
        
        return (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm,
                                                                                                          dim=1)
    
class TrainDataset(Dataset):
    def __init__(self, kg):
        
        
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
                self.triples.append([self.asin2id[asin_1],  self.relation2id[rel], self.asin2id[asin_2]])
        
        self.len = len(self.triples)
            
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.triples[idx], len(self.asin2id)
    
    @staticmethod
    def collate_fn(data):
        positive_sample = [_[0] for _ in data]
        random_entity_ids = np.random.randint(data[0][1], size=len(data))
        
        if random.random() < 0.5:
            negative_sample = [ [l[0][0], l[0][1], random_entity_ids[l_id]]  for l_id, l in enumerate(data)]
        else:
            negative_sample = [ [random_entity_ids[l_id], l[0][1], l[0][2]]  for l_id, l in enumerate(data)]

        return positive_sample, negative_sample

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


                
# Frist we obtain the sentence bert features from the knowledge graph
model = SentenceTransformer('all-MiniLM-L6-v2')
model.cuda()


for domain in ["Electronics", "Clothing_Shoes_and_Jewelry"]:

    graph = nx.read_gpickle(domain + "_00__filtered_kg.gpickle")
    print(domain)
    eventuality_features = {}

    batch_size = 512
    count = 0
    batched_sentences = []
    for u, v, a in tqdm(graph.edges(data=True)):
        count += 1

        if count % batch_size != 0:
            batched_sentences.append(v)
        else:
            batched_sentences.append(v)
            embeddings = model.encode(batched_sentences)
            for sentence, embedding in zip(batched_sentences, embeddings):
                eventuality_features[sentence] = embedding.tolist()

            batched_sentences = []
    
    embeddings = model.encode(batched_sentences)
    for sentence, embedding in zip(batched_sentences, embeddings):
        eventuality_features[sentence] = embedding.tolist()

    with open(domain + "_eventuality_SBERT_feature.json", "w") as fout:
        json.dump(eventuality_features, fout)

        
# The sentence bert feature for each entity pair is in the _eventuality_SBERT_feature.json


# Then we compute the text features for each entity. It is conducted by 
# taking average of the entity-pair semantic in which
# one of the entity is that entity. 
for domain in ["Clothing_Shoes_and_Jewelry", "Electronics"]:
    kg_file = domain + "_00__filtered_kg.gpickle"
    graph = nx.read_gpickle(kg_file)
    
    with open(domain + "_eventuality_SBERT_feature.json", "r") as fin:
        eventuality_features = json.load(fin)
        
    
    all_asins = {}
    for u, v, a in graph.edges(data=True):
        asin_1, asin_2 = u.split("-")
        
        if asin_1 in all_asins:
            all_asins[asin_1].append(eventuality_features[v])
        else:
            all_asins[asin_1] = [eventuality_features[v]]
        
        if asin_2 in all_asins:
            all_asins[asin_2].append(eventuality_features[v])
        else:
            all_asins[asin_2] = [eventuality_features[v]]
        
    
    aggregated_features = {}
    for asin, values_list in all_asins.items():
        values_list_sum = np.mean(values_list, axis=0).tolist()
        aggregated_features[asin] = values_list_sum

    with open(domain + "_text_feature.json", "w") as fout:
        json.dump(aggregated_features, fout)

    

# After these, we compute the graph features by using the vanilla transE on the knowledge graph

for domain in ["Clothing_Shoes_and_Jewelry", "Electronics"]:
 
    graph = nx.read_gpickle(domain + "_00__filtered_kg.gpickle")
    batch_size = 1024

    dataset = TrainDataset(graph)


    train_iterator = SingledirectionalOneShotIterator(DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=TrainDataset.collate_fn
            ))


    model = TransE(len(set([u for u in graph.nodes()])), len(dataset.all_relations), torch.device('cuda'))
    model.cuda()

    len(set([u for u in graph.nodes()]))

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    STEP_LIMIT = 50000


    for step in tqdm(range(STEP_LIMIT)):

        positive_sample, negative_sample= next(train_iterator)

        positive_sample = torch.tensor(positive_sample).cuda()
        negative_sample = torch.tensor(negative_sample).cuda()


        optimizer.zero_grad()

        loss, positive_distance, negative_distance = model(positive_sample, negative_sample)
        loss = loss.mean()

        loss.backward()
        optimizer.step()

        if step % 5000 == 0:
            print(loss)


    feature_dict = {}

    for asin, eid in dataset.asin2id.items():
        feature_dict[asin] = model.entities_emb.weight[eid].detach().cpu().numpy().tolist()

    with open(domain + "_co_buy_transE_feature.json", "w") as fout:
        json.dump(feature_dict, fout)
 



