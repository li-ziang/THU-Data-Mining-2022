
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
data['user'].x = torch.IntTensor(user_info[["age_range","gender"]].to_numpy())
data['user'].id = torch.LongTensor(user_info[["user_id"]].to_numpy()) 

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
NUMBRAND = 8850
NUMCAT = 1700
NUMSELLER = 5000
all_num = NUMSELLER + NUMCAT + NUMBRAND
k_user = user_log[['user_id','cat_id','brand_id', 'seller_id']].to_numpy()
k_user = torch.LongTensor(k_user)
k_user[k_user<0] = 0
user_feature = torch.zeros(data['user'].id.shape[0],all_num, dtype=torch.int32)

for id in tqdm(range(data['user'].id.shape[0])):
    user_id = data['user'].id[id]
    user_id_log = k_user[k_user[:,0]==user_id.item()][:,1:]
    user_feature[id,(2+user_id_log[:,0])]=1
    user_feature[id,(2+NUMCAT+user_id_log[:,1])]=1
    user_feature[id,(2+NUMCAT+NUMBRAND+user_id_log[:,2])]=1

data['user'].x = torch.cat([data['user'].x,user_feature],dim=1).shape