import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import copy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import random
import xgboost as xgb


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'mlp'])
parser.add_argument('--feature', type=int, default=1, choices=[1, 2], help='which feature to use')
parser.add_argument('--new_preprocess', action='store_true', help='whether to run a new preprocess')
parser.add_argument('--no_onehot', action='store_true', help='whether to use onehot representation of age and gender')
args = parser.parse_args()


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, 1))
        self.drop_rate = dropout

    def forward(self, x):
        for layer in self.layers[0:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_rate)
        x = self.layers[-1](x)
        x = torch.sigmoid(x)
        return x


def preprocess1(no_onehot=False):
    print("Using the first preprocess. This might take a few minutes")
    user_log = pd.read_csv('data/taobao/raw/user_log_format1.csv')
    user_log.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
    user_log['brand_id'].fillna(0, inplace=True)

    # user_info的年龄和性别转为one-hot表示
    user_info = pd.read_csv('data/taobao/raw/user_info_format1.csv')
    user_info['age_range'].fillna(0, inplace=True)
    user_info['gender'].fillna(2, inplace=True)
    if not no_onehot:
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

    train_data = data[ data['origin'] == 'train' ].drop(columns=['origin', 'user_id', 'merchant_id'])
    test_data = data[ data['origin'] == 'test' ].drop(columns=['label', 'origin'])
    train_x, train_y = train_data.drop(columns=['label']), train_data['label']
    return train_x.to_numpy(), train_y.to_numpy(), test_data.to_numpy()


def preprocess2(no_onehot=False):
    print("Using the second preprocess. This might take a few minutes")
    train_data = pd.read_csv('data/taobao/raw/train_format1.csv')
    test_data = pd.read_csv('data/taobao/raw/test_format1.csv')
    user_info = pd.read_csv('data/taobao/raw/user_info_format1.csv')
    user_log = pd.read_csv('data/taobao/raw/user_log_format1.csv')
    user_info['age_range'].replace(np.nan, 0.0, inplace=True)
    user_info['gender'].replace(np.nan, 2.0, inplace=True)
    user_log['brand_id'].replace(np.nan, -1.0, inplace=True)

    if not no_onehot:
        age_onehot = np.zeros((len(user_info), 8))
        age_ori = user_info['age_range'].to_numpy()
        del user_info['age_range']
        for i in range(8):
            tmp = np.zeros(8)
            tmp[i] = 1
            age_onehot[age_ori == i] = tmp
            user_info['age_range{}'.format(i)] = age_onehot[:, i]
        
        gender_onehot = np.zeros((len(user_info), 3))
        gender_ori = user_info['gender'].to_numpy()
        del user_info['gender']
        for i in range(3):
            tmp = np.zeros(3)
            tmp[i] = 1
            gender_onehot[gender_ori == i] = tmp
            user_info['gender{}'.format(i)] = gender_onehot[:, i]
    
    train_data = pd.merge(train_data, user_info, on='user_id', how='left')
    test_data = pd.merge(test_data, user_info, on='user_id', how='left')

    # 统计点击数，加购物车数，买的数，收藏数，log数，点击数占log数比例，加购物车数占log数比例，购买数占比，收藏数占比（共9维）
    one_clicks_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["action_type"]]).count()
    one_clicks_temp = one_clicks_temp.reset_index()[["user_id","seller_id","action_type","item_id"]]
    one_clicks_temp.rename(columns={"seller_id":"merchant_id","item_id":"times"},inplace=True)
    one_clicks_temp["clk"] = (one_clicks_temp["action_type"] == 0) * one_clicks_temp["times"]
    one_clicks_temp["cart"] = (one_clicks_temp["action_type"] == 1) * one_clicks_temp["times"]
    one_clicks_temp["buy"] = (one_clicks_temp["action_type"] == 2) * one_clicks_temp["times"]
    one_clicks_temp["fav"] = (one_clicks_temp["action_type"] == 3) * one_clicks_temp["times"]
    four_features = one_clicks_temp.groupby([one_clicks_temp["user_id"],one_clicks_temp["merchant_id"]]).sum().reset_index()
    four_features = four_features.drop(["action_type"], axis=1)
    four_features['clk_rate'] = four_features['clk'] / four_features['times']
    four_features['cart_rate'] = four_features['cart'] / four_features['times']
    four_features['buy_rate'] = four_features['buy'] / four_features['times']
    four_features['fav_rate'] = four_features['fav'] / four_features['times']
    train_data = pd.merge(train_data, four_features, on=['user_id','merchant_id'], how='left')
    test_data = pd.merge(test_data, four_features, on=["user_id","merchant_id"],how="left")

    # 统计浏览的天数，有浏览的时候平均每天浏览数、点赞数、加购物车数、购买数、收藏数（共6维），
    days_tmp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["time_stamp"]])
    days_tmp = days_tmp.count().reset_index()[["user_id","seller_id","time_stamp"]]
    days_tmp = days_tmp.groupby([days_tmp["user_id"],days_tmp["seller_id"]]).count().reset_index()
    days_tmp.rename(columns={"seller_id":"merchant_id","time_stamp":"days"},inplace=True)
    train_data = pd.merge(train_data, days_tmp, on=['user_id','merchant_id'], how='left')
    test_data = pd.merge(test_data, days_tmp, on=["user_id","merchant_id"],how="left")
    for cat in ['clk', 'cart', 'buy', 'fav', 'times']:
        train_data['avg_' + cat] = train_data[cat] / train_data['days']
        test_data['avg_' + cat] = test_data[cat] / test_data['days']

    # 统计浏览的商品的总数、点赞商品总数、加购物车商品总数、购买总数、收藏总数（5维）
    item_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["action_type"],user_log['item_id']]).count()
    item_temp = item_temp.reset_index()[["user_id","seller_id","action_type","item_id"]]
    item_temp = item_temp.groupby([item_temp['user_id'], item_temp['seller_id'], item_temp['action_type']]).count().reset_index()
    item_temp.rename(columns={"seller_id":"merchant_id","item_id":"unique_items"},inplace=True)
    item_temp["clk_item"] = (item_temp["action_type"] == 0) * item_temp["unique_items"]
    item_temp["cart_item"] = (item_temp["action_type"] == 1) * item_temp["unique_items"]
    item_temp["buy_item"] = (item_temp["action_type"] == 2) * item_temp["unique_items"]
    item_temp["fav_item"] = (item_temp["action_type"] == 3) * item_temp["unique_items"]
    item_temp = item_temp.groupby([item_temp["user_id"],item_temp["merchant_id"]]).sum().reset_index()
    item_temp = item_temp.drop(["action_type"], axis=1)
    train_data = pd.merge(train_data, item_temp, on=['user_id','merchant_id'], how='left')
    test_data = pd.merge(test_data, item_temp, on=["user_id","merchant_id"], how="left")

    # 统计浏览的品牌的总数、点赞品牌总数、加购物车品牌总数、购买总数、收藏总数（5维）
    item_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["action_type"],user_log['brand_id']]).count()
    item_temp = item_temp.reset_index()[["user_id","seller_id","action_type","brand_id"]]
    item_temp = item_temp.groupby([item_temp['user_id'], item_temp['seller_id'], item_temp['action_type']]).count().reset_index()
    item_temp.rename(columns={"seller_id":"merchant_id","brand_id":"unique_brands"},inplace=True)
    item_temp["clk_brand"] = (item_temp["action_type"] == 0) * item_temp["unique_brands"]
    item_temp["cart_brand"] = (item_temp["action_type"] == 1) * item_temp["unique_brands"]
    item_temp["buy_brand"] = (item_temp["action_type"] == 2) * item_temp["unique_brands"]
    item_temp["fav_brand"] = (item_temp["action_type"] == 3) * item_temp["unique_brands"]
    item_temp = item_temp.groupby([item_temp["user_id"],item_temp["merchant_id"]]).sum().reset_index()
    item_temp = item_temp.drop(["action_type"], axis=1)
    train_data = pd.merge(train_data, item_temp, on=['user_id','merchant_id'], how='left')
    test_data = pd.merge(test_data, item_temp, on=["user_id","merchant_id"], how="left")

    # 统计浏览的商品类别总数，点赞类别总数、加购物车类别总数、购买类别总数、收藏总数（共5维）
    item_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["action_type"],user_log['cat_id']]).count()
    item_temp = item_temp.reset_index()[["user_id","seller_id","action_type","cat_id"]]
    item_temp = item_temp.groupby([item_temp['user_id'], item_temp['seller_id'], item_temp['action_type']]).count().reset_index()
    item_temp.rename(columns={"seller_id":"merchant_id", "cat_id":"unique_cats"},inplace=True)
    item_temp["clk_cat"] = (item_temp["action_type"] == 0) * item_temp["unique_cats"]
    item_temp["cart_cat"] = (item_temp["action_type"] == 1) * item_temp["unique_cats"]
    item_temp["buy_cat"] = (item_temp["action_type"] == 2) * item_temp["unique_cats"]
    item_temp["fav_cat"] = (item_temp["action_type"] == 3) * item_temp["unique_cats"]
    item_temp = item_temp.groupby([item_temp["user_id"],item_temp["merchant_id"]]).sum().reset_index()
    item_temp = item_temp.drop(["action_type"], axis=1)
    train_data = pd.merge(train_data, item_temp, on=['user_id','merchant_id'], how='left')
    test_data = pd.merge(test_data, item_temp, on=["user_id","merchant_id"], how="left")

    train_y = train_data['label']
    train_x = train_data.drop(columns=['user_id', 'merchant_id', 'label'])
    test_data = test_data.drop(columns=['prob'])
    return train_x.to_numpy(), train_y.to_numpy(), test_data.to_numpy()


# 用于验证集和测试集评估的函数，无梯度
def evaluate(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    predict_list = np.array([])
    label_list = np.array([])

    with torch.no_grad():
        for samples in dataloader:   # 一个个batch去测试
            labels = samples[:, -1]
            inputs = samples[:, 0:-1]
            predict = model(inputs)
            predict_list = np.concatenate((predict_list, predict[:, 0].cpu().numpy()))
            label_list = np.concatenate((label_list, labels.cpu().numpy()))
    
    rocauc = roc_auc_score(label_list, predict_list)
    return rocauc


# 训练+评估的核心函数
def run_mlp(train_x, valid_x, train_y, valid_y, test_data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_y = train_y[:, np.newaxis]
    valid_y = valid_y[:, np.newaxis]
    train_data = np.concatenate((train_x, train_y), axis=1)
    valid_data = np.concatenate((valid_x, valid_y), axis=1)
    train_data = torch.tensor(train_data, device=device, dtype=torch.float32)
    valid_data = torch.tensor(valid_data, device=device, dtype=torch.float32)
    test_data = torch.tensor(test_data, device=device, dtype=torch.float32)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=1000, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False)
    
    model = MLP(test_data.shape[1] - 2, args.hidden_size, args.dropout, args.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    max_val_rocauc = 0
    patience = 0
    best_model = None

    for epoch in range(args.num_epoch):
        model.train()
        losses = []
        for samples in train_dataloader:
            labels = samples[:, -1:]
            inputs = samples[:, 0:-1]
            predict = model(inputs)
            optimizer.zero_grad()
            loss = F.mse_loss(predict, labels)
            loss.backward()
            optimizer.step()      # 更新权重
            losses.append(loss.item())
            
        # 每训练一轮都去验证集上跑一下
        print('epoch:', epoch, ' train loss:', np.mean(losses))
        rocauc = evaluate(model, valid_dataloader)
        print('evaluate rocauc:', rocauc)
        if rocauc > max_val_rocauc:    # 如果有更好的结果，则更新
            max_val_rocauc = rocauc
            best_model = copy.deepcopy(model)
        else:
            patience += 1
            if patience == args.num_epoch / 2:
                break
    
    # 最后用最佳模型测试
    print('max valid rocauc:', max_val_rocauc)
    print('testing')
    outputs_all = torch.tensor([], device=device)
    model.eval()
    with torch.no_grad():
        for inputs in test_dataloader:
            outputs = best_model(inputs[:, 2:])
            outputs_all = torch.concat((outputs_all, outputs))
    
    result = torch.concat((test_data[:, 0:2], outputs_all), dim=1)
    result = pd.DataFrame(columns=['user_id', 'merchant_id', 'prob'], data=result.cpu())
    result['user_id'] = result['user_id'].astype(int)
    result['merchant_id'] = result['merchant_id'].astype(int)
    result.to_csv('sample_submission.csv', index=False)


def run_xgb(train_x, valid_x, train_y, valid_y, test_data):
    train_y = train_y[:, np.newaxis]
    valid_y = valid_y[:, np.newaxis]
    train_data = np.concatenate((train_x, train_y), axis=1)
    valid_data = np.concatenate((valid_x, valid_y), axis=1)

    model = xgb.XGBClassifier(
        max_depth=8,
        n_estimators=1000,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.3,
        eval_metric='auc',
        early_stopping_rounds=10,
        seed=42
    )

    model.fit(
        train_data[:, :-1],
        train_data[:, -1],
        eval_set=[(valid_data[:, :-1], valid_data[:, -1])],
        verbose=True,
    )
    
    test_data = test_data[:, 2:]
    prob = model.predict_proba(test_data)

    data_test = pd.read_csv('data/taobao/raw/test_format1.csv')
    data_test['prob'] = pd.DataFrame(prob[:, 1])
    data_test.to_csv('sample_submission.csv', index=False)


def main():
    if args.model == 'mlp':
        assert args.no_onehot == False, 'MLP model only supports onehot representation'

    try:
        if args.new_preprocess:   # need to preprocess again
            exit()
        train_x = np.load('data/taobao/processed/train_x.npy')
        train_y = np.load('data/taobao/processed/train_y.npy')
        test_data = np.load('data/taobao/processed/test.npy')
    
    except:
        if args.feature == 1:
            train_x, train_y, test_data = preprocess1(args.no_onehot)
        else:
            train_x, train_y, test_data = preprocess2(args.no_onehot)
        np.save('data/taobao/processed/train_x.npy', train_x)
        np.save('data/taobao/processed/train_y.npy', train_y)
        np.save('data/taobao/processed/test.npy', test_data)
    
    idx = list(range(len(train_x)))
    random.shuffle(idx)
    train_x = train_x[idx]
    train_y = train_y[idx]
    valid_x = train_x[int(0.8 * len(train_x)):]
    train_x = train_x[0: int(0.8 * len(train_x))]
    valid_y = train_y[int(0.8 * len(train_y)):]
    train_y = train_y[0: int(0.8 * len(train_y))]
    
    if args.model == 'xgboost':
        run_xgb(train_x, valid_x, train_y, valid_y, test_data)
    else:
        run_mlp(train_x, valid_x, train_y, valid_y, test_data)


if __name__ == '__main__':
    main()
