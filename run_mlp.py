import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import copy
from models.mlp import MLP
from sklearn.metrics import roc_auc_score
import random


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--num_layers', type=int, default=3)
args = parser.parse_args()


def preprocess():
    print("Preprocessing. This might take a few minutes")
    train_data = pd.read_csv('data/taobao/raw/train_format1.csv')
    test_data = pd.read_csv('data/taobao/raw/test_format1.csv')
    user_info = pd.read_csv('data/taobao/raw/user_info_format1.csv')
    user_log = pd.read_csv('data/taobao/raw/user_log_format1.csv')
    user_info['age_range'].replace(np.nan, 0.0, inplace=True)
    user_info['age_range'].replace(8.0, 7.0, inplace=True)
    user_info['gender'].replace(np.nan, 2.0, inplace=True)
    user_log['brand_id'].replace(np.nan, -1.0, inplace=True)

    gender_onehot = np.zeros((len(user_info), 3))
    gender_ori = user_info['gender'].to_numpy()
    del user_info['gender']
    for i in range(3):
        tmp = np.zeros(3)
        tmp[i] = 1
        gender_onehot[gender_ori == i] = tmp
        user_info['gender{}'.format(i)] = gender_onehot[:, i]

    age_onehot = np.zeros((len(user_info), 8))
    age_ori = user_info['age_range'].to_numpy()
    del user_info['age_range']
    for i in range(8):
        tmp = np.zeros(8)
        tmp[i] = 1
        age_onehot[age_ori == i] = tmp
        user_info['age_range{}'.format(i)] = age_onehot[:, i]
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

    train_label = train_data['label'].to_numpy()
    train_data = train_data.drop(['user_id', 'merchant_id', 'label'], axis=1).to_numpy()
    test_data = test_data.drop(['prob'], axis=1).to_numpy()
    train_label = train_label[:, np.newaxis]
    train_data = np.concatenate((train_data, train_label), axis=1)
    np.save('./data/taobao/processed/mlp_train.npy', train_data)
    np.save('./data/taobao/processed/mlp_test.npy', test_data)
    print(train_data.shape, test_data.shape)


# 用于验证集和测试集评估的函数，无梯度
def evaluate(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    predict_list = np.array([])
    label_list = np.array([])

    with torch.no_grad():
        for samples in dataloader:   # 一个个batch去测试
            labels = samples[:, 41]
            inputs = samples[:, 0:41]
            predict = model(inputs)
            predict_list = np.concatenate((predict_list, predict[:, 0].cpu().numpy()))
            label_list = np.concatenate((label_list, labels.cpu().numpy()))
    
    rocauc = roc_auc_score(label_list, predict_list)
    return rocauc


# 训练+评估的核心函数
def train_and_eval(train_data, valid_data, test_data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            labels = samples[:, 41:42]
            inputs = samples[:, 0:41]
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
    
    outputs_all = torch.concat((test_data[:, 0:2], outputs_all), dim=1)
    outputs_all = pd.DataFrame(columns=['user_id', 'merchant_id', 'prob'], data=outputs_all.cpu())
    outputs_all.to_csv('output/result.csv', index=False)


def main():
    try:
        train_data = np.load('./data/taobao/processed/mlp_train.npy')
        test_data = np.load('./data/taobao/processed/mlp_test.npy')
    except:
        preprocess()
        train_data = np.load('./data/taobao/processed/mlp_train.npy')
        test_data = np.load('./data/taobao/processed/mlp_test.npy')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = torch.tensor(train_data, device=device, dtype=torch.float32)
    idx = list(range(len(train_data)))
    random.shuffle(idx)
    train_data = train_data[idx]

    test_data = torch.tensor(test_data, device=device, dtype=torch.float32)
    valid_data = train_data[int(0.8 * len(train_data)):]
    train_data = train_data[: int(0.8 * len(train_data))]
    train_and_eval(train_data, valid_data, test_data)


if __name__ == '__main__':
    main()
