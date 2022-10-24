import pandas as pd
import catboost as cb
from preprocess import params
from preprocess import get_treshholds
from preprocess import make_full_pred
from sklearn.metrics import recall_score
data_dir = './data'
X_train = pd.read_csv(data_dir + '/X_train.csv')
X_test = pd.read_csv(data_dir + '/X_test.csv')
y_train = pd.read_csv(data_dir + '/y_train.csv')
y_test = pd.read_csv(data_dir + '/y_test.csv')
X = pd.read_csv(data_dir + '/X.csv')
y = pd.read_csv(data_dir + '/y.csv')
X_t = pd.read_csv(data_dir + 'X_t.csv')
real_test = pd.read_csv(data_dir, '/test.csv')
pool_train = cb.Pool(X_train, y_train, cat_features=cat)
pool_test = cb.Pool(X_test, y_test, cat_features=cat)
submit_pool_train = cb.Pool(X, y, cat_features=cat)
submit_pool_test = cb.Pool(X_t, cat_features=cat)
clf_local = cb.CatBoostClassifier(**params)
clf_local.fit(pool_train)
pred_proba = clf_local.predict_proba(pool_test)
pred_job = [0 if i[0] < 0.7 else 1 for i in pred_proba]
pred_div = [0 if i[1] < 0.6 else 1 for i in pred_proba]
pred = make_full_pred(pred_job, pred_div)
full_y_test = make_full_pred(list(y_test['job']), list(y_test['division']))
print("Recall score:", recall_score(y_test['job'], pred_job, average='macro'))
print("Recall score:", recall_score(y_test['division'], pred_div, average='macro'))
print("Recall score:", recall_score(full_y_test, pred, average='macro'))
clf_sub = cb.CatBoostClassifier(**params)
clf_sub.fit(submit_pool_train)
pred_proba = clf_sub.predict_proba(submit_pool_test)
real_test = pd.read_csv('/content/test.csv')
min_t1, min_t2, min_error = get_tresholds(pred_proba, real_test)
pred_job = [0 if i[0] < min_t1 else 1 for i in pred_proba]
pred_div = [0 if i[1] < min_t2 else 1 for i in pred_proba]
pred = make_full_pred(pred_job, pred_div)
real_test['type'] = pred
real_test.to_csv('submit_якутск.csv', index=False)
