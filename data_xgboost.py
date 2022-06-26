import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split

def generate_data(data_path='./data/taobao/raw', test_size=0.2):
    if os.path.exists('./.cache'):
        X_train = pickle.load(open('.cache/x_train.pkl', 'rb'))
        X_valid = pickle.load(open('.cache/x_valid.pkl', 'rb'))
        y_train = pickle.load(open('.cache/y_train.pkl', 'rb'))
        y_valid = pickle.load(open('.cache/y_valid.pkl', 'rb'))
        test_data = pickle.load(open('.cache/test_data.pkl', 'rb'))
        return X_train, X_valid, y_train, y_valid, test_data
    os.mkdir('.cache')
    user_log = pd.read_csv(f'{data_path}/user_log_format1.csv')
    user_log.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
    user_log['brand_id'].fillna(0, inplace=True)

    user_info = pd.read_csv(f'{data_path}/user_info_format1.csv')
    user_info['age_range'].fillna(0, inplace=True)
    user_info['gender'].fillna(2, inplace=True)
    user_info = pd.concat((
        user_info, 
        pd.get_dummies(user_info['age_range'].astype('int8'), prefix='a'), 
        pd.get_dummies(user_info['gender'].astype('int8'), prefix='g')
    ), axis=1)
    user_info.drop(columns=['age_range', 'gender'], inplace=True)

    data_train = pd.read_csv(f'{data_path}/train_format1.csv')
    data_test = pd.read_csv(f'{data_path}/test_format1.csv')
    data_test.drop(columns=['prob'], inplace=True)
    data_train['origin'] = 'train'
    data_test['origin'] = 'test'
    data = pd.concat([data_train, data_test], ignore_index=True)
    data_train.drop(columns=['origin'], inplace=True)
    data_test.drop(columns=['origin'], inplace=True)
    data = data.merge(user_info, on='user_id', how='left')

    groups = user_log.groupby(['user_id'])
    data = data.merge(groups[['item_id', 'cat_id', 'merchant_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'user_items',
        'cat_id': 'user_cats',
        'merchant_id': 'user_merchants',
        'brand_id': 'user_brands',
    }), on='user_id', how='left')
    data = data.merge(groups['action_type'].value_counts().unstack().reset_index().rename(columns={
        0: 'user_clk', 
        1: 'user_cart', 
        2: 'user_buy', 
        3: 'user_fav',
    }).fillna(0).astype('int'), on='user_id', how='left')

    groups = user_log.groupby(['merchant_id'])
    data = data.merge(groups[['item_id', 'cat_id', 'user_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'merchant_items',
        'cat_id': 'merchant_cats',
        'user_id': 'merchant_users',
        'brand_id': 'merchant_brands',
    }), on='merchant_id', how='left')
    data = data.merge(groups['action_type'].value_counts().unstack().reset_index().rename(columns={
        0: 'merchant_clk',
        1: 'merchant_cart',
        2: 'merchant_buy',
        3: 'merchant_fav',
    }).fillna(0).astype('int'), on='merchant_id', how='left')
    data = data.merge(
        data_train[ data_train['label'] == 0 ].groupby(['merchant_id']).size().reset_index().rename(columns={0: 'merchant_label0'}), 
        on='merchant_id', how='left')
    data = data.merge(
        data_train[ data_train['label'] == 1 ].groupby(['merchant_id']).size().reset_index().rename(columns={0: 'merchant_label1'}), 
        on='merchant_id', how='left') # 加入后提升明显

    groups = user_log.groupby(['user_id', 'merchant_id'])
    data = data.merge(groups[['item_id', 'cat_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id':'items',
        'cat_id':'cats',
        'brand_id':'brands'
    }), on=['user_id', 'merchant_id'], how='left')
    data = data.merge(groups['action_type'].value_counts().unstack().reset_index().rename(columns={
        0: 'clk',
        1: 'cart',
        2: 'buy',
        3: 'fav',
    }), on=['user_id', 'merchant_id'], how='left')

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

    train_data = data[ data['origin'] == 'train' ].drop(columns=['origin'])
    test_data = data[ data['origin'] == 'test' ].drop(columns=['label', 'origin'])
    train_X, train_y = train_data.drop(columns=['label']), train_data['label']
    X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=test_size)

    pickle.dump(X_train, open('.cache/x_train.pkl', 'wb'))
    pickle.dump(X_valid, open('.cache/x_valid.pkl', 'wb'))
    pickle.dump(y_train, open('.cache/y_train.pkl', 'wb'))
    pickle.dump(y_valid, open('.cache/y_valid.pkl', 'wb'))
    pickle.dump(test_data, open('.cache/test_data.pkl', 'wb'))
    return X_train, X_valid, y_train, y_valid, test_data