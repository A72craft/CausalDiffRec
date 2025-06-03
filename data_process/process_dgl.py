import pandas as pd
import dgl
import torch
import os
import numpy as np

os.makedirs('../dataset/review', exist_ok=True)

# Read data
train_data = pd.read_csv('../dataset/txt/training.txt', sep=' ', header=None, names=['user_id', 'item_id'])
test_data = pd.read_csv('../dataset/txt/testing.txt', sep=' ', header=None, names=['user_id', 'item_id'])

unique_users = train_data['user_id'].unique()
user_id_map = {old: new for new, old in enumerate(unique_users)}
train_data['user_id'] = train_data['user_id'].map(user_id_map)
test_data['user_id'] = test_data['user_id'].map(user_id_map).fillna(-1)  # Mark unknown users

#items
unique_items = train_data['item_id'].unique()
item_id_map = {old: new for new, old in enumerate(unique_items)}
train_data['item_id'] = train_data['item_id'].map(item_id_map)
test_data['item_id'] = test_data['item_id'].map(item_id_map).fillna(-1)  # Mark unknown items

#offset items to come after all users
num_users = len(unique_users)
train_data['item_id'] += num_users
test_data['item_id'] += num_users

#remove any interactions with unknown users/items
test_data = test_data[(test_data['user_id'] >= 0) & (test_data['item_id'] >= num_users)]

#calculate actual number of nodes needed
num_nodes = num_users + len(unique_items)

print("\n=== Corrected Statistics ===")
print(f"Unique users: {num_users}")
print(f"Unique items: {len(unique_items)}")
print(f"Total nodes needed: {num_nodes}")
print(f"  - Users: 0-{num_users-1}")
print(f"  - Items: {num_users}-{num_nodes-1}")

#convert to numpy arrays before creating graph
train_users = train_data['user_id'].to_numpy()
train_items = train_data['item_id'].to_numpy()
test_users = test_data['user_id'].to_numpy()
test_items = test_data['item_id'].to_numpy()

#create graphs
train_g = dgl.graph((train_users, train_items), num_nodes=num_nodes)
test_g = dgl.graph((test_users, test_items), num_nodes=num_nodes)

#add features(random)
feature_dim = 128
train_g.ndata['feat'] = torch.clamp(torch.randn(num_nodes, feature_dim), -1, 1)
test_g.ndata['feat'] = train_g.ndata['feat'].clone()

#verify
print("\n=== Graph Verification ===")
print(f"Train graph: {train_g.number_of_nodes()} nodes, {train_g.number_of_edges()} edges")
print(f"Test graph: {test_g.number_of_nodes()} nodes, {test_g.number_of_edges()} edges")

# Save graphs
dgl.save_graphs("../dataset/review/review_train_data.bin", train_g)
dgl.save_graphs("../dataset/review/review_test_data.bin", test_g)
