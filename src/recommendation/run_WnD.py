from math import sqrt

import argparse
import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from torch.nn import MSELoss

import pickle
import json


# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))



class WnD_model(nn.Module):

    def __init__(self, n_users, n_items, embed_size, features, dropout_rate = 0.6):
        super(WnD_model, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_size = embed_size

        self.user_embeddings = nn.Embedding(n_users, embed_size)
        self.item_embeddings = nn.Embedding(n_items, embed_size)

        self.features = features


        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_1 = nn.Linear(embed_size * 2
                                 + features.weight.shape[-1]
                                 , 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 1)

        self.wide_layer = nn.Linear(embed_size * 2
                                    + features.weight.shape[-1], 1)

    def forward(self, user_id, item_id):

        user_embeddings = self.user_embeddings(user_id)
        item_embeddings = self.item_embeddings(item_id)

        features = self.features(item_id)
  

        all_embeddings = torch.cat((user_embeddings, item_embeddings, features), dim=-1)
        relu = nn.ReLU()

        layer_1_output = relu(self.layer_1(self.dropout(all_embeddings)))
        layer_2_output = relu(self.layer_2(self.dropout(layer_1_output)))
        layer_3_output = relu(self.layer_3(self.dropout(layer_2_output))).reshape(-1)

        wide_output = self.wide_layer(self.dropout(all_embeddings)).reshape(-1)

        return layer_3_output + wide_output

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The training Neural Colaborative Filtering')

  
    parser.add_argument('--total_steps', default=70000, type=int, help='train log every xx steps')
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    
    parser.add_argument('-dr', '--dropout_rate', default=0.6, type=float)
    
    
    parser.add_argument('-d', '--domain', type=str, required=True)
    parser.add_argument('-r', '--ratio',type=str, required=True)

    

    args = parser.parse_args()
    
    print("start training")
    
    
    n_epoch = args.total_steps
    
    
    # domain = "Electronics"
    domain = args.domain
    
    with open(domain + "_matrix_train.pickle", "rb") as fin:
        train_matrix = pickle.load(fin)

    with open(domain + "_matrix_valid.pickle", "rb") as fin:
        valid_matrix = pickle.load(fin)

    with open(domain + "_matrix_test.pickle", "rb") as fin:
        test_matrix = pickle.load(fin)
        
    test_results = []
    
    f = open("test_record_WnD_" + domain + "_" + args.ratio + "_filtered.txt", "w")

    for run_count in range(5):
        
        
        
        
        # Build User/Item vocabulary
        user_set = set([l[1] for l in train_matrix])
        business_set = set([l[0] for l in train_matrix])

        user_vocab = dict(zip(user_set, range(1, len(user_set) + 1)))
        user_vocab['unk'] = 0
        n_users = len(user_vocab)
        business_vocab = dict(zip(business_set, range(1, len(business_set) + 1)))
        business_vocab['unk'] = 0
        n_items = len(business_vocab)

        tr_users = list(map(lambda x: user_vocab[x] if x in user_vocab else 0, [ l[1] for l in train_matrix ]))
        tr_items = list(map(lambda x: business_vocab[x] if x in business_vocab else 0, [ l[0] for l in train_matrix ]))
        tr_ratings = [l[2] for l in train_matrix]

        tr_users = np.array(tr_users)
        tr_items = np.array(tr_items)
        tr_ratings = np.array(tr_ratings)

        val_users = list(map(lambda x: user_vocab[x] if x in user_vocab else 0, [ l[1] for l in valid_matrix ]))
        val_items = list(map(lambda x: business_vocab[x] if x in business_vocab else 0, [ l[0] for l in valid_matrix ]))
        val_ratings = [l[2] for l in valid_matrix]

        val_users = np.array(val_users)
        val_items = np.array(val_items)
        val_ratings = np.array(val_ratings)

        te_users = list(map(lambda x: user_vocab[x] if x in user_vocab else 0, [ l[1] for l in test_matrix ]))
        te_items = list(map(lambda x: business_vocab[x] if x in business_vocab else 0, [ l[0] for l in test_matrix ]))
        te_ratings = [l[2] for l in test_matrix]

        te_users = np.array(te_users)
        te_items = np.array(te_items)
        te_ratings = np.array(te_ratings)
        
        
        # Clothing_Shoes_and_Jewelry_07_filtered_transE_feature.json
        
        with open(domain + "_"+ args.ratio + "_filtered_transE_feature.json", "r") as fin:
            transE_feature_dict = json.load(fin)

        transE_feature_matrix = np.zeros((n_items, len( list(transE_feature_dict.values())[0] )))

        count = 0
        for asin, feat in transE_feature_dict.items():

            if asin in business_vocab:
                count += 1
                transE_feature_matrix[business_vocab[asin]] = feat


        TransE_feature_embeddings = nn.Embedding.from_pretrained(torch.tensor(
            transE_feature_matrix, dtype=torch.float32
        )).cuda()
        for param in TransE_feature_embeddings.parameters():
            param.requires_grad = False


    
 

        model_dot = WnD_model(n_users, n_items, 50, TransE_feature_embeddings, dropout_rate=args.dropout_rate)
        model_dot.cuda()
        model_dot.cuda()
        

        tr_users = torch.tensor(tr_users).cuda()
        tr_items = torch.tensor(tr_items).cuda()
        tr_ratings = torch.tensor(tr_ratings, dtype=torch.float32).cuda()

        val_users = torch.tensor(val_users).cuda()
        val_items = torch.tensor(val_items).cuda()

        te_users = torch.tensor(te_users).cuda()
        te_items = torch.tensor(te_items).cuda()


        optimizer = optim.Adam(model_dot.parameters(), lr=0.0001)
        loss_fn = MSELoss()

        pbar = tqdm(range(n_epoch))
        
        epoch_counter = 0
        batch_size = 4096
        
        validation_metrics = []
        testing_metrics = []

        for epoch in pbar:
            model_dot.train()
            optimizer.zero_grad()


            batched_ids = np.random.randint( len(tr_users), size=batch_size)
            batched_users = tr_users [batched_ids]
            batched_items = tr_items [batched_ids]

            output = model_dot(batched_users, batched_items)
            loss = loss_fn(output, tr_ratings[batched_ids])
            loss.backward()
            optimizer.step()

            pbar.set_description("Loss %.5f" % loss.cpu().detach().item())


            if epoch_counter % 500 == 0:
                model_dot.eval()
                val_output = model_dot(val_users, val_items)
                val_output = val_output.cpu().detach()
                
                te_output = model_dot(te_users, te_items)
                te_output = te_output.cpu().detach()
                
                valid_metric = rmse(val_output, val_ratings)
                test_metric = rmse(te_output, te_ratings)
                
                validation_metrics.append(valid_metric)
                testing_metrics.append(test_metric)
                
                
        
                print("VALID RMSE: %s \n" %  rmse(val_output, val_ratings))
                print("TEST RMSE: %s \n" % rmse(te_output, te_ratings))

            epoch_counter += 1
        min_index = np.argmin(validation_metrics)
        best_test = testing_metrics[min_index]
        
        test_results.append(best_test)
    
    test_average = np.mean(test_results)
    test_std = np.std(test_results)
    f.write("Test results: \n")
    f.write("\n".join([str(val) for val in test_results]))
    f.write("\n mean: ")
    f.write(str(test_average))
    f.write("std: ")
    f.write(str(test_std))
    
    f.close()