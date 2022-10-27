import pandas as pd
from preprocess import *
data_dir = './data'
df_train = pd.read_csv(data_dir + "/train.csv")
df_test = pd.read_csv(data_dir + "/test.csv")
df_skud = pd.read_csv(data_dir + '/SKUD.csv')
df_calls = pd.read_csv(data_dir + "/Calls.csv")
df_tasks = pd.read_csv(data_dir + '/Tasks.csv')
df_connection_time = pd.read_csv(data_dir + "/ConnectionTime.csv")
df_working_day = pd.read_csv(data_dir + '/WorkingDay.csv')
df_ed = pd.read_csv(data_dir + "/Education.csv")
df_network = pd.read_csv(data_dir + '/TimenNetwork.csv')
SEED = set_seed()
df_train['division'] = df_train['type'].apply(change_division)
df_train['job'] = df_train['type'].apply(change_job)
outliers = []
grpb = df_network.groupby('Вых/Будни')['monitor_Time']
for i in ['Будни', "Выходные дни"]:
  outliers.append(df_network.loc[( grpb.median()[i] - grpb.mad()[i] > df_network['monitor_Time']) | (df_network['monitor_Time'] > grpb.median()[i] + grpb.mad()[i])])
df_network['is_outlier_Будни'] = [check4index(i, outliers[0]) for i in range(len(df_network))]
df_network['is_outlier_Выходные'] = [check4index(i, outliers[1]) for i in range(len(df_network))]
df_network['monitor_Time'] = round(df_network['monitor_Time']/60, 3)
df_N = pd.DataFrame()
df_N['Выходные_outliers'] = df_network.groupby('id')['is_outlier_Выходные'].sum()
df_N[[f'monitor_Time_{i}' for i in stats]] = df_network.groupby('id')['monitor_Time'].agg(stats)
tmp = df_network.groupby(['id'])['monitor_Time'].agg(['min', 'max'])
df_N['frequency_monitor_Time'] = df_N['monitor_Time_count'] / (tmp['max'] - tmp['min'] + 1)
df_WD = pd.DataFrame()
idx = outlier_detection(df_working_day, 'activeTime').index
df_working_day['is_outlier_activeTime'] = [check4index(i, idx) for i in range(len(df_working_day))]
idx = outlier_detection(df_working_day, 'monitorTime').index
df_working_day['is_outlier_monitorTime'] = [check4index(i, idx) for i in range(len(df_working_day))]
df_WD['activeTime_outliers'] = df_working_day.groupby('id')['is_outlier_activeTime'].sum()
df_WD['monitorTime_outliers'] = df_working_day.groupby('id')['is_outlier_monitorTime'].sum()
df_working_day = df_working_day.drop(combined_fdrop(df_working_day, 'activeTime', 'monitorTime')
df_WD[[f'activeTime_{i}' for i in stats]] = df_working_day.groupby('id')['activeTime'].agg(stats)
df_WD[[f'monitorTime_{i}' for i in stats]] = df_working_day.groupby('id')['monitorTime'].agg(stats)
tmp = {'activeTime':df_working_day.groupby('id')['activeTime'].count().agg(['min', 'max']), 'monitorTime':df_working_day.groupby('id')['monitorTime'].count().agg(['min', 'max'])}
df_WD['frequency_activeTime'] = df_WD['activeTime_count'] / (tmp['activeTime']['max'] - tmp['activeTime']['min'] + 1)
df_WD['frequency_monitorTime'] = df_WD['monitorTime_count'] / (tmp['monitorTime']['max'] - tmp['monitorTime']['min'] + 1)
df_tasks = df_tasks.drop(fdrop(df_tasks, 'Просрочено, дней'))
df_tasks['Статус по просрочке'] = df_tasks['Статус по просрочке'].apply(bin_cat('С нарушением срока'))
dummies = pd.get_dummies(df_tasks['Вид документа'])
dummies['id'] = df_tasks['id']
df_T = pd.DataFrame()
df_T['Количество задач'] = df_tasks.groupby('id')['ID задачи'].nunique()
df_T['Просрочено дней'] = df_tasks.groupby('id')["Просрочено, дней"].sum()
df_T['Статус по просрочке'] = df_tasks.groupby('id')['Статус по просрочке'].sum()
df_T['mode_task_type'] = df_tasks.groupby('id')['Вид документа'].agg(pd.Series.mode)
df_T[[f'type04tasks_{i}' for i in dummies.columns[:-1]]] = dummies.groupby('id').agg('sum')/dummies.groupby('id').agg('count')
df_T['count'] = dummies.groupby('id')["Акт"].agg('count')
df_tasks['Состояние задания'] = df_tasks['Состояние задания'].apply(bin_cat('Делегировано'))
df_T['Количество делегированных задач'] = df_tasks.groupby('id')['Состояние задания'].sum()
df_skud['Длительность раб.дня без обеда'] = df_skud['Длительность раб.дня без обеда'].apply(to_float)
df_skud['Длительность общая'] = df_skud['Длительность общая'].apply(to_float)
df_S = pd.DataFrame()
idx = outlier_detection(df_skud, 'Длительность общая').index
df_skud['is_outlier_Длительность общая'] = [check4index(i, idx) for i in range(len(df_skud))]
df_S['Длительность_outliers'] = df_skud.groupby('id')['is_outlier_Длительность общая'].sum()
df_skud = df_skud.drop(combined_fdrop(df_skud, 'Длительность раб.дня без обеда','Длительность общая')) 
df_S['Длительность_std'] = df_skud.groupby('id')["Длительность общая"].std()
df_S['Длительность_med'] = df_skud.groupby('id')["Длительность общая"].median()
df_calls['Вид учета времени'] = df_calls['Вид учета времени'].apply(bin_cat("Выходные дни"))
df_calls['CallTime'] = df_calls['CallTime'].apply(to_float)
df_c = pd.DataFrame()
idx = (outlier_detection(df_calls, 'NumberOfCalls').index & outlier_detection(df_calls, 'CallTime').index)
df_calls['is_outlier_Calls'] = [check4index(i, idx) for i in range(len(df_calls))]
df_c['is_outlier_Calls'] = df_calls.groupby('id')['is_outlier_Calls'].sum()
df_calls = df_calls.drop(combined_fdrop(df_calls, 'NumberOfCalls', 'CallTime'))
df_c['c_T'] = df_calls.groupby('id')['Вид учета времени'].sum()
df_calls['InOut'] = df_calls['InOut'].apply(bin_cat('InOut'))
df_c[['s_C', 'c_C']] = df_calls.groupby('id')['InOut'].agg(['sum', 'count'])
df_c[[f'CallTime_{i}' for i in stats]] = df_calls.groupby('id')['CallTime'].agg(stats)
df_ed['Вид образования'] = df_ed['Вид образования'].apply(change_ed)
df_E = pd.DataFrame()
df_E['count_ed'] = df_ed.groupby('id')['Табельный номер руководителя'].count()
df_E['ed_type_1'] = df_ed.groupby('id')['Вид образования'].apply(lambda x: x.unique()[-1])
df_E['ed_type_2'] = df_ed.groupby('id')['Вид образования'].apply(lambda x: x.unique()[-2] if x.nunique() > 1 else 0)
# df_E['ed_type_3'] = df_ed.groupby('id')['Вид образования'].apply(lambda x: x.unique()[-3] if x.nunique() > 2 else 0)
df_E['Руководитель'] = df_ed.groupby('id')['Табельный номер руководителя'].apply(lambda x: x.unique()[0])
for i in range(4):
  df_E[f'ed_spec_{i}'] = df_ed.groupby('id')['Специальность'].apply(lambda x: x.unique()[-(i + 1)] if x.nunique() > i else 0)
df_connection_time['Время опоздания'] = df_connection_time['Время опоздания'].apply(lambda x: float(str(x).replace(',', '.')))
df_connection_time_res = df_connection_time.drop(df_connection_time.loc[df_connection_time['Вых/Будни'] == 'Выходные дни'].index)[['id', 'Признак опоздания']]
df_connection_time_res = df_connection_time_res[df_connection_time_res["Признак опоздания"].notna()]
tmp = df_connection_time_res["Признак опоздания"].notna()
df_connection_time_res =  df_connection_time.groupby('id')['Время опоздания'].agg(stats)
df_connection_time_res = df_connection_time_res.rename(columns={"sum": "Время опоздания_sum", 
                                                                'median':'Время опоздания_median',
                                                                'count':'Время опоздания_count',
                                                                'std':'Время опоздания_std'})
df_N = make_transformed_cols(df_E, df_N, cols2, cols1_N)
df_connection_time_res = make_transformed_cols(df_E, df_connection_time_res, cols2, cols1_C)
dfs = [df_train, df_S, df_c, df_T, df_WD, df_N, df_connection_time_res]
df_full = make_full_df(dfs, df_E)
df_train = pd.merge(df_train, df_full, on='id', how='left')
df_test = pd.merge(df_test, df_full, on='id', how='left')
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)
lst_drop = get_lst_drop(df_train.columns)
lst_test_drop = get_lst_drop(df_test.columns, is_test=True)
cats = get_cats(df_train.columns)
df_train[cat] = df_train[cat].astype('string') 
df_test[cat] = df_test[cat].astype('string') 
tmp=df_train.drop(lst_drop[:-3], axis=1).drop_duplicates()
X = tmp.drop(['type', 'job', 'division'], axis=1)
X_t = df_test.drop([i.strip() for i in lst_test_drop], axis = 1)
y = tmp[['job', 'division',]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.to_csv(data_dir + '/X_train.csv')
X_test.to_csv(data_dir + '/X_test.csv')
y_train.to_csv(data_dir + '/y_train.csv')
y_test.to_csv(data_dir + '/y_test.csv')
X.to_csv(data_dir + '/X.csv')
y.to_csv(data_dir + '/y.csv')
X_t.to_csv(data_dir + '/X_t.csv')
df_full.to_csv(data_ir + '/full_data.csv')
