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
        '''
        user_log = pd.read_csv('data/taobao/raw/user_log_format1.csv')
        user_log.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
        user_log['brand_id'].fillna(0, inplace=True)

        # user_info的年龄和性别转为one-hot表示
        user_info = pd.read_csv('data/taobao/raw/user_info_format1.csv')
        user_info['age_range'].fillna(0, inplace=True)
        user_info['gender'].fillna(2, inplace=True)
        user_info = pd.concat((
            user_info,
            pd.get_dummies(user_info['age_range'].astype('int8'), prefix='a'),
            pd.get_dummies(user_info['gender'].astype('int8'), prefix='g')
        ), axis=1)
        user_info.drop(columns=['age_range', 'gender'], inplace=True)

        # 把data_train和data_test拼起来变成data
        data_train = pd.read_csv('data/taobao/raw/train_format1.csv')
        data_test = pd.read_csv('data/taobao/raw/test_format1.csv')
        data_test.drop(columns=['prob'], inplace=True)
        data_train['origin'] = 'train'
        data_test['origin'] = 'test'
        data = pd.concat([data_train, data_test], ignore_index=True)
        data_train.drop(columns=['origin'], inplace=True)
        data_test.drop(columns=['origin'], inplace=True)
        data = data.merge(user_info, on='user_id', how='left')
        
        # 统计用户的整体信息：总商品数、总类别数、总商家数、总品牌数
        groups = user_log.groupby(['user_id'])
        data = data.merge(groups[['item_id', 'cat_id', 'merchant_id', 'brand_id']].nunique().reset_index().rename(columns={
            'item_id': 'user_items',
            'cat_id': 'user_cats',
            'merchant_id': 'user_merchants',
            'brand_id': 'user_brands',
        }), on='user_id', how='left')
        # 统计用户的总浏览类信息：点击、加购物车、购买、收藏
        data = data.merge(groups['action_type'].value_counts().unstack().reset_index().rename(columns={
            0: 'user_clk', 
            1: 'user_cart', 
            2: 'user_buy', 
            3: 'user_fav',
        }).fillna(0).astype('int'), on='user_id', how='left')

        # 统计商家的整体信息：商品数、类别数、用户数、品牌数
        groups = user_log.groupby(['merchant_id'])
        data = data.merge(groups[['item_id', 'cat_id', 'user_id', 'brand_id']].nunique().reset_index().rename(columns={
            'item_id': 'merchant_items',
            'cat_id': 'merchant_cats',
            'user_id': 'merchant_users',
            'brand_id': 'merchant_brands',
        }), on='merchant_id', how='left')
        # 统计商家的总浏览类信息：点击、加购物车、购买、收藏
        data = data.merge(groups['action_type'].value_counts().unstack().reset_index().rename(columns={
            0: 'merchant_clk',
            1: 'merchant_cart',
            2: 'merchant_buy',
            3: 'merchant_fav',
        }).fillna(0).astype('int'), on='merchant_id', how='left')
        # 统计商家收获的复购情况：被多少用户复购过
        data = data.merge(
            data_train[ data_train['label'] == 0 ].groupby(['merchant_id']).size().reset_index().rename(columns={0: 'merchant_label0'}), 
            on='merchant_id', how='left')
        data = data.merge(
            data_train[ data_train['label'] == 1 ].groupby(['merchant_id']).size().reset_index().rename(columns={0: 'merchant_label1'}), 
            on='merchant_id', how='left') # 加入后提升明显

        # 该用户在该商家的信息：购买商品数、类别数、品牌数
        groups = user_log.groupby(['user_id', 'merchant_id'])
        data = data.merge(groups[['item_id', 'cat_id', 'brand_id']].nunique().reset_index().rename(columns={
            'item_id':'items',
            'cat_id':'cats',
            'brand_id':'brands'
        }), on=['user_id', 'merchant_id'], how='left')
        # 该用户在该商家的浏览类信息：点击、加购物车、购买、收藏
        data = data.merge(groups['action_type'].value_counts().unstack().reset_index().rename(columns={
            0: 'clk',
            1: 'cart',
            2: 'buy',
            3: 'fav',
        }), on=['user_id', 'merchant_id'], how='left')

        # 各种行为的比例
        data['r1'] = data['user_buy'] / data['user_clk']
        data['r2'] = data['merchant_buy'] / data['merchant_clk']
        data['r3'] = data['buy'] / data['clk']
        data['r4'] = data['user_cart'] / data['user_clk']
        data['r5'] = data['merchant_cart'] / data['merchant_clk']
        data['r6'] = data['cart'] / data['clk']
        data['r7'] = data['user_fav'] / data['user_clk']
        data['r8'] = data['merchant_fav'] / data['merchant_clk']
        data['r9'] = data['fav'] / data['clk']
        data.fillna(0, inplace=True)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        '''

        data = pd.read_csv('data_processed.csv')

        data['user_id'] -= 1
        all_user = np.unique(data['user_id'].to_numpy())
        assert all_user.shape[0] == 424170
        all_merchant = np.unique(data['merchant_id'].to_numpy())
        assert all_merchant.shape[0] == 1994

        merchant_index_to_val = np.zeros(5000, dtype=np.int64) - 1
        for i, merchant in enumerate(all_merchant):
            merchant_index_to_val[merchant] = i
        data['merchant_id'] = merchant_index_to_val[data['merchant_id'].to_numpy().astype(int)]
        
        hetero_data = HeteroData()
        train_data = data[ data['origin'] == 'train' ][['user_id', 'merchant_id', 'label']]
        test_data = data[ data['origin'] == 'test' ][['user_id', 'merchant_id']]
        hetero_data['user'].train = torch.LongTensor(train_data.to_numpy())
        hetero_data['user'].test = torch.LongTensor(test_data.to_numpy())

        hetero_data['user'].x = torch.FloatTensor(data.iloc[all_user][['user_items', 'user_cats', 'user_merchants', 'user_brands',
            'user_clk', 'user_cart', 'user_buy', 'user_fav', 'r1', 'r4', 'r7', 'a_0', 'a_1', 'a_2', 'a_3', 'a_4',
            'a_5', 'a_6', 'a_7', 'a_8', 'g_0', 'g_1', 'g_2']].to_numpy())
        hetero_data['seller'].x = torch.FloatTensor(data.iloc[all_merchant][['merchant_items', 'merchant_cats', 'merchant_users',
            'merchant_brands', 'merchant_clk', 'merchant_cart', 'merchant_buy','merchant_fav', 
            'merchant_label0', 'merchant_label1', 'r2', 'r5', 'r8']].to_numpy())

        hetero_data['user','buy','seller'].edge_index = torch.LongTensor(data[['user_id', 'merchant_id']].to_numpy().T)
        hetero_data['user','buy','seller'].edge_attr = torch.FloatTensor(data[['items', 'cats', 'brands', 'clk', 'cart',
            'buy', 'fav', 'r3', 'r6', 'r9']].to_numpy())

        hetero_data['seller','bought by','user'].edge_index = torch.LongTensor(data[['merchant_id', 'user_id']].to_numpy().T)
        hetero_data['seller','bought by','user'].edge_attr = torch.FloatTensor(data[['items', 'cats', 'brands', 'clk', 'cart',
            'buy', 'fav', 'r3', 'r6', 'r9']].to_numpy())

        torch.save(self.collate([hetero_data]), self.processed_paths[0])
