
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
import os
import os.path as osp
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

user_info = pd.read_csv('/data/ziang/data-mining/data/data_format1/user_info_format1.csv')
data = HeteroData()
user_x = user_info[["age_range","gender"]]
user_x[np.isnan(user_x)]=-1
data['user'].x = torch.LongTensor(user_x.to_numpy())
data['user'].id = torch.LongTensor(user_info[["user_id"]].to_numpy())-1

root = "/data/ziang/data-mining/data/data_format1/user_log_format1.csv"
user_log = pd.read_csv(root)
print("read done")
num_users = np.unique(user_log["user_id"].to_numpy())
num_max_user = user_log["user_id"].to_numpy().max()
num_min_user = user_log["user_id"].to_numpy().min()
print(num_users)

num_sellers = np.unique(user_log["seller_id"].to_numpy())
num_max_sellers = user_log["seller_id"].to_numpy().max()
num_min_sellers = user_log["seller_id"].to_numpy().min()
print(num_sellers)

'''
user_log.head()
Out[158]: 
   user_id  item_id  cat_id  seller_id  brand_id  time_stamp  action_type
0   328862   323294     833       2882    2661.0         829            0
1   328862   844400    1271       2882    2661.0         829            0
2   328862   575153    1271       2882    2661.0         829            0
3   328862   996875    1271       2882    2661.0         829            0
4   328862  1086186    1271       1253    1049.0         829            0

item_id 太大了,不用
'cat_id','brand_id', 'seller_id'可以用作用户的特征 用 many hot向量
brand 有 8844个, cat 有1658个 seller有4995个
'''

k_user = user_log[['user_id','cat_id']].to_numpy()
NUM_USERS = 424170 # 从 1到 424170
NUM_CATS = 1671
user_additional_feature = np.zeros([NUM_USERS,NUM_CATS+1])
for i in tqdm(range(k_user.shape[0])):
    user_additional_feature[k_user[i,0]-1,k_user[i,1]] +=1

user_additional_feature = torch.LongTensor(user_additional_feature)
data['user'].id = data['user'].id.reshape(-1)
sort_result = data['user'].id.sort()
id_map = sort_result[1]
data['user'].id = sort_result[0].reshape(-1)
data['user'].x = data['user'].x[id_map.reshape(-1)]
data['user'].x = torch.cat([data['user'].x,user_additional_feature],dim=1)


k_seller = user_log[['seller_id','cat_id']].to_numpy()
NUM_SELLERS = 4995
seller_additional_feature = np.zeros([NUM_SELLERS,NUM_CATS+1])
for i in tqdm(range(k_seller.shape[0])):
    seller_additional_feature[k_seller[i,0]-1,k_seller[i,1]] +=1
data['seller'].id = torch.arange(NUM_SELLERS,dtype=torch.long)
data['seller'].x = torch.LongTensor(seller_additional_feature)

# data['user','buy','seller']

all_log = user_log.to_numpy()

data['user','buy','seller'].edge_index = torch.LongTensor(all_log[:,[0,4]].T)
data['user','buy','seller'].edge_attr = torch.LongTensor(all_log[:,[1,2,3,5,6]])