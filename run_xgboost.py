from data_xgboost import generate_data
from sklearn.metrics import roc_auc_score
import xgboost
import pandas as pd

X_train, X_valid, y_train, y_valid, test_data = generate_data()

model = xgboost.XGBClassifier(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    eval_metric='auc',
    early_stopping_rounds=3,
    seed=42,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
)

prob = model.predict_proba(test_data)
data_test = pd.read_csv('./data/taobao/raw/test_format1.csv')
data_test['prob'] = pd.DataFrame(prob[:, 1])
data_test.to_csv('sample_submission.csv', index=False)
