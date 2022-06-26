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
from typing import Callable, List, Optional


class TaobaoDataset(InMemoryDataset):
    def __init__(self, root: str, preprocess: Optional[str] = None,
            transform: Optional[Callable] = None, 
            pre_transform: Optional[Callable] = None):
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        assert self.preprocess in [None, 'metapath2vec', 'transe']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        pass

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'taobao', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'taobao', 'processed')

    def download(self):
        pass

    @property
    def raw_file_names(self) -> List[str]:
        file_names = [
            'test_format1.csv','train_format1.csv',
            'user_info_format1.csv','user_log_format1.csv',
        ]

        if self.preprocess is not None:
            file_names += [f'mag_{self.preprocess}_emb.pt']

        return file_names

    @property
    def processed_file_names(self) -> str:
        if self.preprocess is not None:
            return f'data_{self.preprocess}.pt'
        else:
            return 'data.pt'

    def __repr__(self) -> str:
        return 'taobao'

    def process(self):
        data = HeteroData()
        train_set = pd.read_csv(osp.join(self.root,'taobao/raw/train_format1.csv'))
        test_set = pd.read_csv(osp.join(self.root,'taobao/raw/test_format1.csv'))
        
        train_user = train_set['user_id'].to_numpy()
        test_user = test_set['user_id'].to_numpy()
        all_user = np.unique(np.concatenate((train_user,test_user)))
        assert all_user.shape[0] == 424170

        train_merchant = train_set['merchant_id'].to_numpy()
        test_merchant = test_set['merchant_id'].to_numpy()
        all_merchant = np.unique(np.concatenate((train_merchant,test_merchant)))
        assert all_merchant.shape[0] == 1994

        merchant_index_to_val = np.zeros(5000,dtype=np.int64)-1
        for i,merchant in enumerate(all_merchant):
            merchant_index_to_val[merchant] = i
        train_set['user_id']-=1
        test_set['user_id']-=1
        train_set = train_set.to_numpy()
        test_set = test_set.to_numpy()[:,:2]
        train_set[:,1] = merchant_index_to_val[train_set[:,1].astype(int)]
        test_set[:,1] = merchant_index_to_val[test_set[:,1].astype(int)]
        data['user'].train = torch.LongTensor(train_set)
        data['user'].test = torch.LongTensor(test_set)

        user_info = pd.read_csv(osp.join(self.root, 'taobao/raw/user_info_format1.csv'))
        # user_info.head()
        user_x = user_info[["age_range","gender"]].to_numpy()
        user_x[:,0][np.isnan(user_x[:,0])] = 0
        user_x[:,1][np.isnan(user_x[:,1])] = 2
        data['user'].x1 = torch.LongTensor(user_x)
        data['user'].id = torch.LongTensor(user_info[["user_id"]].to_numpy())-1

        user_log = pd.read_csv(osp.join(self.root,'taobao/raw/user_log_format1.csv'))
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

        k_user = user_log[['user_id','cat_id','brand_id']].to_numpy()
        k_user[np.isnan(k_user)] = 0
        k_user = k_user.astype(np.int64)
        NUM_USERS = 424170 # 从 1到 424170
        NUM_CATS = 1671
        NUM_BRANDS = 8844
        # pandas
        user_additional_feature = np.zeros([NUM_USERS,NUM_CATS+1])
        user_additional_feature2 = np.zeros([NUM_USERS,NUM_BRANDS+1])
        for i in tqdm(range(k_user.shape[0])):
            user_additional_feature[k_user[i,0]-1,k_user[i,1]] +=1
            user_additional_feature2[k_user[i,0]-1,k_user[i,2]] +=1

        user_additional_feature = torch.FloatTensor(user_additional_feature)
        user_additional_feature2 = torch.FloatTensor(user_additional_feature2)

        data['user'].id = data['user'].id.reshape(-1)
        sort_result = data['user'].id.sort()
        id_map = sort_result[1]
        data['user'].id = sort_result[0].reshape(-1)
        data['user'].x1 = data['user'].x1[id_map.reshape(-1)]
        # correct order

        values, counts = np.unique(user_log["cat_id"].to_numpy(), return_counts=True)
        value_ind = values[counts.argsort()[-128:][::-1]]
        # data['user'].x = data['user'].x[:,value_ind]
        values, counts = np.unique(user_log["brand_id"].to_numpy(), return_counts=True)
        value_ind2 = values[counts.argsort()[-512:][::-1]]
        data['user'].x = torch.cat((user_additional_feature[:,value_ind],user_additional_feature2[:,value_ind2]),1)

        k_seller = user_log[['seller_id','cat_id']].to_numpy()
        NUM_SELLERS = 5000
        seller_additional_feature = np.zeros([NUM_SELLERS,NUM_CATS+1])
        for i in tqdm(range(k_seller.shape[0])):
            seller_additional_feature[k_seller[i,0],k_seller[i,1]] +=1
            #!!!!  kai !
            # seller_additional_feature[k_seller[i,0]-1,k_seller[i,1]] +=1
        data['seller'].id = torch.arange(NUM_SELLERS,dtype=torch.long)
        data['seller'].x = torch.FloatTensor(seller_additional_feature[:,:])
        data['seller'].id = data['seller'].id[all_merchant]
        data['seller'].x = data['seller'].x[all_merchant]
        # all_merchant = [1,2,3]
        # [[0,0],[1,1],[3,2],[2,3]]
        # 加特征：单个seller在训练集中label为1和为0的次数

        print("{} sellers in total".format(data['seller'].x.shape[0]))

        # data['user','buy','seller']

        all_log = user_log.to_numpy()
        mask = np.isin(all_log[:,3].astype(int),all_merchant)
        all_log = all_log[mask]
        edge_index = all_log[:,[0,3]].T
        where_nan = np.isnan(edge_index).sum(axis=0,dtype=bool)
        where_not_nan = np.logical_not(where_nan)
        #  "edge_index" 里面有nan
        #  np.isnan(all_log[:,[0,4]]).sum()=91015
        all_log = all_log[where_not_nan,:]
        u, indices = np.unique(all_log[:,[0,3]],return_index=True, axis=0)
        all_log = all_log[indices]
        all_log = all_log.astype(int)
        all_log[:,0] -= 1
        all_log[:,3] = merchant_index_to_val[all_log[:,3]]
        data['user','buy','seller'].edge_index = torch.LongTensor(all_log[:,[0,3]].T)
        data['user','buy','seller'].edge_attr = torch.FloatTensor(all_log[:,[1,2,4,5,6]])

        data['seller','bought by','user'].edge_index = torch.LongTensor(all_log[:,[3,0]].T)
        data['seller','bought by','user'].edge_attr = torch.FloatTensor(all_log[:,[1,2,4,5,6]])

        train_set_add = train_set[:,0]*10000+train_set[:,1]
        test_set_add = test_set[:,0]*10000+test_set[:,1]
        all_log_add = all_log[:,0]*10000+all_log[:,3]

        assert((train_set_add<0).sum()==0)
        assert((test_set_add<0).sum()==0)

        xsorted = np.argsort(all_log_add)
        train_set_add = np.searchsorted(all_log_add[xsorted], train_set_add)
        train_indices = xsorted[train_set_add]
        test_set_add = np.searchsorted(all_log_add[xsorted], test_set_add)
        test_indices = xsorted[test_set_add]
        data['user'].train_ind = torch.LongTensor(train_indices)
        data['user'].test_ind = torch.LongTensor(test_indices)

        torch.save(self.collate([data]), self.processed_paths[0])
