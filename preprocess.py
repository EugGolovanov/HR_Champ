import pandas as pd
import random
import numpy as np
change_division = lambda x: 1 if x == 1 or x == 2 else 0
change_job = lambda x: 1 if x == 0 or x == 2 else 0 
outlier_detection = lambda df, col: df.loc[(df[col].median() - df[col].mad() > df[col]) | (df[col] > df[col].median() + df[col].mad())]
fdrop = lambda df, col: outlier_detection(df, col).index
combined_fdrop = lambda df, col1, col2: (fdrop(df, col1) | fdrop(df, col2)) 
stats = ['sum', 'median', 'count', 'std']
bin_cat = lambda y: lambda x: 1 if x==y else 0 
to_float = lambda x: float(x.replace(',', '.'))
check4index = lambda i, idx: 0 if i in idx else 1
get_education = lambda x, i: x.unique()[-(i + 1)] if x.nunique() > i else 0)
cols2 = ['Руководитель', 'ed_type_1', 'ed_spec_1']
cols1_N = ['monitor_Time_median', 'Выходные_outliers', 'frequency_monitor_Time', 'monitor_Time_count']
cols1_C = ['Время опоздания_sum', 'Время опоздания_median', 'Время опоздания_count', 'Время опоздания_std']
def outliers(df, col, is_many=False):
    if not is_many:
        idx = fdrop(df, col)
        new_df = pd.DataFrame()
        df[f'is_outlier_{col}'] = [check4index(i, idx) for i in range(len(df))]
        new_df[f'{col}_outliers'] = df.groupby('id')[f'is_outlier_{col}'].sum()
    else:
        col1, col2 = col
        idx = combined_fdrop(df, col1, col2)
        df[f'is_outlier_{col}'] = [check4index(i, idx) for i in range(len(df))]
        new_df[f'{col}_outliers'] = df.groupby('id')[f'is_outlier_{col}'].sum()
    return new_df, df
def change_ed(text):
  if text == 'Высшее образование - бакалавриат' or text == 'Высшее образование - специалитет, магистратура':
    return 'Высшее образование'
  elif text == 'Начальное профессиональное образование' or text == 'Профессиональное обучение' or text == 'Неполное высшее образование' or text == 'Дополнительное профессиональное образование':
    return 'Среднее профессиональное образование'
  elif text == 'Переподготовка':
    return 'Повышение квалификации'
  elif text == 'Среднее (полное) общее образование' or text == 'Основное общее образование' or text == 'Начальное общее образование':
    return 'Среднее общее образование'
  elif text == 'Послевузовское образование':
    return 'Аспирантура'
  else: return text
def make_transformed_cols(df_2, df_1, cols2, cols1)
    tmp = df_2[cols2]
    df_1 = pd.merge(df_1, df_2, on='id', how='left')
    for i in range(len(cols2)):
        for j in range(len(cols1)):
            df_1[f"Mean_{cols1[j]}_{cols2[i]}"] = df_1.groupby(cols[i])[cols1[j]].transform("mean")
    df_1 = df_1.drop(cols2, axis=1)
    return df_1
def make_full_data(dfs, df_E):
    df_full = df_E
    for df in dfs:
       df_full = pd.merge(df_full, df, on='id', how='left')
    for i in range(len(dfs)):
       dfs[i]['Руководитель_x'] = dfs[i].index
    df_full['Руководитель_x'] = df_full['Руководитель']
    df_full = df_full.drop('Руководитель', axis=1)
    for df in dfs:
       df_full = pd.merge(df_full, df, on='Руководитель_x', how='left')
    return df_full
def make_lst_drop(cols, is_test=False):
    lst_drop = []
    for i in cols:
       if col[0] + col[1] == 'id' or col[0] == 'Р':
           lst_drop.append(i)
    lst_drop.append('type')
    if not is_test:
        lst_drop.append('job');lst_drop.append('division')
    return lst_drop
def get_cats(cols):
    cat = []
    for i in df_train.columns:
        if i[0] == 'e' or i == 'mode_task_type_y' or i == 'mode_task_type_x':
def make_full_pred(pred_job, pred_div):
  full_pred = []
  assert len(pred_job)==len(pred_div)
  for i in range(len(pred_div)):
    tmp = {'div':pred_div[i], 'job':pred_job[i]}
    if tmp['job'] == 1 and tmp['div'] == 0:
      full_pred.append(0)
    elif tmp['job'] == 0 and tmp['div'] == 1:
      full_pred.append(1)
    elif tmp['job'] == 1 and tmp['div'] == 1:
      full_pred.append(2)
    else: full_pred.append(3)
  return full_pred
            cat.append(i)
    return cat
def get_tresholds(pred_proba, real_test):
  min_error = 1
  min_t1 = 1
  min_t2 = 1
  for i in range(100, 160):
    for j in range(100, 160):
      t1 = i/160
      t2 = j/160
      pred_job = [0 if i[0] < t1 else 1 for i in pred_proba]
      pred_div = [0 if i[1] < t2 else 1 for i in pred_proba]
      pred = make_full_pred(pred_job, pred_div)
      real_test['type'] = pred
      local_tmp = (real_test['type'].value_counts() / real_test['type'].count())
      error = abs(local_tmp.median() - local_tmp.mean()) + local_tmp.median() +  local_tmp.std() + local_tmp.max() - local_tmp.min()
      if error < min_error:
        min_t1 = t1
        min_t2 = t2
        min_error = error
  return min_t1, min_t2, min_error
params = {                  # 'task_type':'GPU',
                            'iterations':3000,
                            'depth':7,
                            'random_state':SEED,
                            'learning_rate':0.01,
                            'eval_metric':'HammingLoss',
                            'loss_function':'MultiCrossEntropy'}
def set_seed():
    SEED = 0xCAFEC0DE
    random.seed(SEED)
    np.random.seed(SEED)
    return SEED
