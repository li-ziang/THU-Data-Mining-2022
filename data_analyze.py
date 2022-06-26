import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    train_data = pd.read_csv('data/taobao/raw/train_format1.csv')
    test_data = pd.read_csv('data/taobao/raw/test_format1.csv')
    user_info = pd.read_csv('data/taobao/raw/user_info_format1.csv')
    user_log = pd.read_csv('data/taobao/raw/user_log_format1.csv')
    user_info['age_range'].replace(np.nan, 0.0, inplace=True)
    user_info['gender'].replace(np.nan, 2.0, inplace=True)
    train_data = pd.merge(train_data, user_info, on='user_id', how='left')
    test_data = pd.merge(test_data, user_info, on='user_id', how='left')

    # 整体情况
    print('#train data:', len(train_data), '; #label:', train_data['label'].sum(), 
        '; label rate:', train_data['label'].sum() / len(train_data))
    print(test_data.shape, train_data.shape)
    print(user_info.shape, user_log.shape)  

    # 统计不同年龄段的数据分布
    print(len(user_info[user_info['age_range'] == 1]))
    print(len(user_info[user_info['age_range'] == 2]))
    print(len(user_info[user_info['age_range'] == 3]))
    print(len(user_info[user_info['age_range'] == 4]))
    print(len(user_info[user_info['age_range'] == 5]))
    print(len(user_info[user_info['age_range'] == 6]))
    print(len(user_info[user_info['age_range'] == 7]))
    print(len(user_info[user_info['age_range'] == 8]))
    print(len(user_info[user_info['age_range'] == 0]))
    x = np.array(["NULL","<18","18-24","25-29","30-34","35-39","40-49",">=50"])
    y = np.array([user_info[user_info['age_range'] == 0]['age_range'].count(),
                user_info[user_info['age_range'] == 1]['age_range'].count(),
                user_info[user_info['age_range'] == 2]['age_range'].count(),
                user_info[user_info['age_range'] == 3]['age_range'].count(),
                user_info[user_info['age_range'] == 4]['age_range'].count(),
                user_info[user_info['age_range'] == 5]['age_range'].count(),
                user_info[user_info['age_range'] == 6]['age_range'].count(),
                user_info[user_info['age_range'] == 7]['age_range'].count() + user_info[user_info['age_range'] == 8]['age_range'].count()])
    plt.bar(x, y, label='number')
    plt.legend()
    plt.title('age distribution')
    plt.savefig('age distribution.png')

    # 统计不同年龄段的label情况
    print('\ntrain data age:')
    for i in range(9):
        tmp = train_data[train_data['age_range'] == i]
        print('age range:', i, '; cnt:', len(tmp), '; #label:', tmp['label'].sum(), 
            '; label rate:', tmp['label'].sum() / len(tmp))
    print('\ntest data age:')
    for i in range(9):
        tmp = test_data[test_data['age_range'] == i]
        print('age range:', i, '; cnt:', len(tmp))
    
    # 统计不同性别的label情况
    print(len(user_info[user_info['gender'] == 0]))
    print(len(user_info[user_info['gender'] == 1]))
    print(len(user_info[user_info['gender'] == 2]))
    print('\ntrain data gender:')
    for i in range(3):
        tmp = train_data[train_data['gender'] == i]
        print('gender:', i, '; cnt:', len(tmp), '; #label:', tmp['label'].sum(), 
            '; label rate:', tmp['label'].sum() / len(tmp))
    print('\ntest data gender:')
    for i in range(3):
        tmp = test_data[test_data['gender'] == i]
        print('gender:', i, '; cnt:', len(tmp))

    # 统计交易信息
    user_log_clk = user_log[user_log['action_type'] == 0]
    user_log_add = user_log[user_log['action_type'] == 1]
    user_log_buy = user_log[user_log['action_type'] == 2]
    user_log_save = user_log[user_log['action_type'] == 3]

    user_log_clk = user_log_clk.groupby([user_log_clk['user_id'], user_log_clk['seller_id']]).count().reset_index()[['user_id', 'seller_id', 'item_id']]
    user_log_clk.rename(columns={'seller_id': 'merchant_id', 'item_id': 'clk_cnt'}, inplace=True)
    user_log_add = user_log_add.groupby([user_log_add['user_id'], user_log_add['seller_id']]).count().reset_index()[['user_id', 'seller_id', 'item_id']]
    user_log_add.rename(columns={'seller_id': 'merchant_id', 'item_id': 'add_cnt'}, inplace=True)
    user_log_buy = user_log_buy.groupby([user_log_buy['user_id'], user_log_buy['seller_id']]).count().reset_index()[['user_id', 'seller_id', 'item_id']]
    user_log_buy.rename(columns={'seller_id': 'merchant_id', 'item_id': 'buy_cnt'}, inplace=True)
    user_log_save = user_log_save.groupby([user_log_save['user_id'], user_log_save['seller_id']]).count().reset_index()[['user_id', 'seller_id', 'item_id']]
    user_log_save.rename(columns={'seller_id': 'merchant_id', 'item_id': 'save_cnt'}, inplace=True)
    user_log_tot = user_log.groupby([user_log['user_id'], user_log['seller_id']]).count().reset_index()[['user_id', 'seller_id', 'item_id']]
    user_log_tot.rename(columns={'seller_id': 'merchant_id', 'item_id': 'log_cnt'}, inplace=True)

    train_data = pd.merge(train_data, user_log_clk, on=['user_id', 'merchant_id'], how='left')
    train_data = pd.merge(train_data, user_log_add, on=['user_id', 'merchant_id'], how='left')
    train_data = pd.merge(train_data, user_log_buy, on=['user_id', 'merchant_id'], how='left')
    train_data = pd.merge(train_data, user_log_save, on=['user_id', 'merchant_id'], how='left')
    train_data = pd.merge(train_data, user_log_tot, on=['user_id', 'merchant_id'], how='left')
    
    print('\ntrain data label = 1:')
    tmp = train_data[train_data['label'] == 1]
    print('平均点击:', tmp['clk_cnt'].sum() / len(tmp), '至少点一次的占比:', (tmp['clk_cnt']>0).sum() / len(tmp))
    print('平均加购物车:', tmp['add_cnt'].sum() / len(tmp), '至少加一次的占比:', (tmp['add_cnt']>0).sum() / len(tmp))
    print('平均买:', tmp['buy_cnt'].sum() / len(tmp), '至少买一次的占比:', (tmp['buy_cnt']>0).sum() / len(tmp))
    print('平均收藏:', tmp['save_cnt'].sum() / len(tmp), '至少收藏一次的占比:', (tmp['save_cnt']>0).sum() / len(tmp))
    print('平均log数:', tmp['log_cnt'].sum() / len(tmp), '至少一次记录的占比:', (tmp['log_cnt']>0).sum() / len(tmp))

    print('\ntrain data label = 0:')
    tmp = train_data[train_data['label'] == 0]
    print('平均点击:', tmp['clk_cnt'].sum() / len(tmp), '至少点一次的占比:', (tmp['clk_cnt']>0).sum() / len(tmp))
    print('平均加购物车:', tmp['add_cnt'].sum() / len(tmp), '至少加一次的占比:', (tmp['add_cnt']>0).sum() / len(tmp))
    print('平均买:', tmp['buy_cnt'].sum() / len(tmp), '至少买一次的占比:', (tmp['buy_cnt']>0).sum() / len(tmp))
    print('平均收藏:', tmp['save_cnt'].sum() / len(tmp), '至少收藏一次的占比:', (tmp['save_cnt']>0).sum() / len(tmp))
    print('平均log数:', tmp['log_cnt'].sum() / len(tmp), '至少一次记录的占比:', (tmp['log_cnt']>0).sum() / len(tmp))


if __name__ == '__main__':
    main()
