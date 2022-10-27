import pandas as pd
import catboost as cb
from preprocess import *
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
df_full = pd.read_csv(data_dir + '/full_data.csv')
df_full['id'] = df_full.index
df_full['type'] = 3
df_full['job'] = 0
df_full['division'] = 0
df_train = pd.read_csv(data_dir + "/train.csv")
df_test = pd.read_csv(data_dir + "/test.csv")
df_train = pd.merge(df_train, df_full, on='id', how='left')
df_test = pd.merge(df_test, df_full, on='id', how='left')
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)
lst_drop = get_lst_drop(df_train.columns)
lst_test_drop = get_lst_drop(df_test.columns, is_test=True)
cats = get_cats(df_train.columns)
df_train[cat] = df_train[cat].astype('string') 
df_test[cat] = df_test[cat].astype('string') 
aux_tmp = df_full.drop(lst_drop, axis=1)
aux_tmp = aux_tmp.fillna(0)
aux_tmp[cat] = aux_tmp[cat].astype('string')
X_auxiliary = cb.Pool(aux_tmp, cat_features=cat)
id_auxiliary = df_full.index
pred_proba_auxiliary = clf_sub.predict_proba(X_auxiliary)
# N = 400 # топ 400 минимальных по вероятности 3его класса 
tmp = pd.DataFrame()
f = lambda x, k: [i[k] for i in x]
tmp['proba_1'] = f(pred_proba_auxiliary, 0) 
tmp['proba_2'] = f(pred_proba_auxiliary, 1) 
tmp['proba_1'] = tmp['proba_1'] < min_t1-0.1
tmp['proba_2'] = tmp['proba_2'] < min_t2-0.1
idxs = tmp.loc[tmp['proba_1'] & tmp['proba_2']].index
auxiliary = df_full.iloc[idxs]
auxiliary['type'] = 3
auxiliary['job'] = 0
auxiliary['division'] = 0
X_train = pd.concat([df_train, auxiliary], axis=0)
X_train['division'] = X_train['type'].apply(change_division)
X_train['job'] = X_train['type'].apply(change_job)
y_train = X_train[['job', 'division']].fillna(0)
aux_tmp = X_train.drop(lst_drop, axis=1)
aux_tmp = aux_tmp.fillna(0)
aux_tmp[cat] = aux_tmp[cat].astype('string')
pool_train = cb.Pool(aux_tmp, y_train, cat_features=cat)
params = {                  # 'task_type':'GPU',
                            'iterations':1200,
                            'depth':7,
                            'random_state':SEED,
                            'learning_rate':0.01,
                            'eval_metric':'HammingLoss',
                            'loss_function':'MultiCrossEntropy'}
clf_with_aux = cb.CatBoostClassifier(**params)
clf_with_aux.fit(pool_train)
pred_proba = clf_with_aux.predict_proba(submit_pool_test)
min_t1, min_t2, min_error = get_tresholds(pred_proba)
pred_job = [0 if i[0] < min_t1 else 1 for i in pred_proba]
pred_div = [0 if i[1] < min_t2 else 1 for i in pred_proba]
pred = make_full_pred(pred_job, pred_div)
real_test['type'] = pred
real_test.to_csv('submit_якутск.csv', index=False)
