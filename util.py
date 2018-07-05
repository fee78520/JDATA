#encoding=utf8
import pandas as pd
from sklearn import preprocessing

def load_data():
    sku_info = pd.read_csv(r'../data/jdata_sku_basic_info.csv')
    user_action = pd.read_csv(r'../data/jdata_user_action.csv')
    user_info = pd.read_csv(r'../data/jdata_user_basic_info.csv')
    user_comment = pd.read_csv(r'../data/jdata_user_comment_score.csv')
    user_order = pd.read_csv(r'../data/jdata_user_order.csv')
    
    user_action['a_date'] = pd.to_datetime(user_action['a_date'])
    user_order['o_date'] = pd.to_datetime(user_order['o_date'])
    user_comment['c_date'] = pd.to_datetime(user_comment['comment_create_tm'])
    user_comment = user_comment.drop('comment_create_tm',axis=1)
    
    user_action['a_year'] = user_action['a_date'].dt.year
    user_action['a_month'] = user_action['a_date'].dt.month
    user_action['a_day'] = user_action['a_date'].dt.day
    
    user_order['o_year'] = user_order['o_date'].dt.year
    user_order['o_month'] = user_order['o_date'].dt.month
    user_order['o_day'] = user_order['o_date'].dt.day
    
    user_comment['c_year'] = user_comment['c_date'].dt.year
    user_comment['c_month'] = user_comment['c_date'].dt.month
    user_comment['c_day'] = user_comment['c_date'].dt.day
    
    #把user_order,user_comment,sku_info,user_info连接在一起组成order_comment表
    order = user_order.merge(sku_info,on='sku_id',how='left')
    order = order.merge(user_info,on='user_id',how='left')
    order = order.merge(user_comment,on=['user_id','o_id'],how='left')
    
    #把user_action,user_info,sku_info连接在一起组成user_action表
    action = user_action.merge(sku_info,on='sku_id',how='left')
    action = action.merge(user_info,on='user_id',how='left')

    return order,action

def encode_onehot(df,column_name):
    feature_df=pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1),feature_df], axis=1)
    return all

def encode_count(df,column_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df

def merge_count(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_nunique(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_median(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_mean(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_sum(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_max(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_min(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_std(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def feat_count(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_nunique(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_nunique" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_mean(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_std(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_median(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_max(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_min(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_sum(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_sum" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_var(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].var()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_var" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def last_day(df, data , time, value, name=""):
    data = data.drop_duplicates(["user_id"], keep="last").reset_index(drop=True)
    # print(data)
    data = data[['user_id', value]]
    data[value] = data[value].apply(lambda x: (time - x).days)
    data.columns = ['user_id'] +  [name]
    df = pd.DataFrame(df)
    data = pd.DataFrame(data)
    df = df.merge(data, on='user_id', how="left").fillna(300)
    return df

def action_last(df, data, time, value, type, name=''):
    data = data[data['a_type'] == type]
    data = data.drop_duplicates(["user_id"], keep="last")[['user_id', value]].reset_index(drop=True)
    data[value] = data[value].apply(lambda x: (time - x).days)
    data.columns = ['user_id'] + [name]
    # print(data)
    data = pd.DataFrame(data)
    df = df.merge(data, on='user_id', how="left").fillna(300)
    return df

def action_not_buy_101(df, order, action, a, month):
    order = order[['user_id', 'sku_id', 'o_date', 'o_id']]
    action = action[['user_id', 'sku_id', 'cate', 'a_date', 'a_num', 'a_type']]
    order.rename(columns={'o_date': 'a_date'}, inplace=True)
    new_action = pd.merge(action, order, on=['user_id', 'sku_id', 'a_date'], how='left')
    new_action = new_action.sort_values(['user_id', 'a_date', 'sku_id'])
    # new_action = pd.merge(new_action, sku, on='sku_id', how='left')
    new_action = new_action[(new_action['cate'] == 101)]
    new_action['month'] = new_action['a_date'].apply(lambda x: x.month)
    new_action = new_action[new_action['month'] == month]
    new_action = new_action[['user_id', 'sku_id', 'a_date', 'a_num', 'a_type', 'o_id']]
    new_action.fillna(0, inplace=True)
    new_order = new_action[new_action['o_id'] != 0]
    new_order = new_order.drop_duplicates(['user_id'], keep='last').reset_index(drop=True)
    # print(new_order)
    new_order = new_order[['user_id', 'a_date']]

    new_order.rename(columns={'a_date': 'o_date_max'}, inplace=True)
    new_action = pd.merge(new_action, new_order, on='user_id', how='left')
    # print(new_action)
    new_action = new_action[new_action['a_date'] > new_action['o_date_max']]
    new_action = new_action[['user_id', 'a_num', 'a_type']]
    new_action_1 = new_action[new_action['a_type'] == 1].reset_index(drop=True)
    new_action_2 = new_action[new_action['a_type'] == 2].reset_index(drop=True)
    new_action_1 = new_action_1[['user_id', 'a_num']].groupby('user_id').sum().reset_index()
    new_action_2 = new_action_2[['user_id', 'a_num']].groupby('user_id').sum().reset_index()
    new_action_1.rename(columns={'a_num': 'a_num_1_101'}, inplace=True)
    # print(new_action_1)
    new_action_2.rename(columns={'a_num': 'a_num_2_101'}, inplace=True)
    df = df.merge(new_action_1, on='user_id', how="left").fillna(0)
    df = df.merge(new_action_2, on='user_id', how="left").fillna(0)
    df['a_num_1_101'].apply(lambda x: 1 if x>0 else 0)
    df['a_num_2_101'].apply(lambda x: 1 if x>0 else 0)
    return df


def action_not_buy_30(df, order, action, a, month):
    order = order[['user_id', 'sku_id', 'o_date', 'o_id']]
    action = action[['user_id', 'sku_id', 'cate', 'a_date', 'a_num', 'a_type']]
    order.rename(columns={'o_date': 'a_date'}, inplace=True)
    new_action = pd.merge(action, order, on=['user_id', 'sku_id', 'a_date'], how='left')
    new_action = new_action.sort_values(['user_id', 'a_date', 'sku_id'])
    # new_action = pd.merge(new_action, sku, on='sku_id', how='left')
    new_action = new_action[(new_action['cate'] == 30)]
    new_action['month'] = new_action['a_date'].apply(lambda x: x.month)
    new_action = new_action[new_action['month'] == month]
    new_action = new_action[['user_id', 'sku_id', 'a_date', 'a_num', 'a_type', 'o_id']]
    new_action.fillna(0, inplace=True)
    new_order = new_action[new_action['o_id'] != 0]
    new_order = new_order.drop_duplicates(['user_id'], keep='last').reset_index(drop=True)
    # print(new_order)
    new_order = new_order[['user_id', 'a_date']]
    # '/' \
    # '' \
    # ''
    new_order.rename(columns={'a_date': 'o_date_max'}, inplace=True)
    new_action = pd.merge(new_action, new_order, on='user_id', how='left')
    # print(new_action)
    new_action = new_action[new_action['a_date'] > new_action['o_date_max']]
    new_action = new_action[['user_id', 'a_num', 'a_type']]
    new_action_1 = new_action[new_action['a_type'] == 1].reset_index(drop=True)
    new_action_2 = new_action[new_action['a_type'] == 2].reset_index(drop=True)
    new_action_1 = new_action_1[['user_id', 'a_num']].groupby('user_id').sum().reset_index()
    new_action_2 = new_action_2[['user_id', 'a_num']].groupby('user_id').sum().reset_index()
    new_action_1.rename(columns={'a_num': 'a_num_1_30'}, inplace=True)
    new_action_2.rename(columns={'a_num': 'a_num_2_30'}, inplace=True)
    df = df.merge(new_action_1, on='user_id', how="left").fillna(0)
    df = df.merge(new_action_2, on='user_id', how="left").fillna(0)
    df['a_num_1_30'].apply(lambda x: 1 if x > 0 else 0)
    df['a_num_2_30'].apply(lambda x: 1 if x > 0 else 0)
    return df


# 根据平均购买时间落在当月月份
def guanxing(df, order, endtime, name=""):
    # order = order.merge(sku, on='sku_id', how='left')
    order = order[(order['cate'] == 101)]
    label_time = pd.to_datetime(endtime)
    order = order[order['o_date']<label_time].reset_index(drop=True)
    order = order[['user_id', 'sku_id', 'cate', 'o_date', 'o_sku_num']]
    order_num = pd.DataFrame(order.groupby(('user_id'))['o_sku_num'].sum()).reset_index()

    order = order.sort_values(['user_id', 'cate', 'o_date', 'sku_id', 'o_sku_num'])
    order_first = order.drop_duplicates(['user_id',],keep='first').reset_index(drop=True)[['user_id', 'o_date']]

    order_last = order.drop_duplicates(['user_id'],keep='last').reset_index(drop=True)[['user_id', 'o_date', 'o_sku_num']]

    order_mean = pd.merge(order_first, order_last, on='user_id')
    order_mean['mean_time'] = (order_mean['o_date_y'] - order_mean['o_date_x'])
    order_mean = order_mean[['user_id', 'mean_time']]
    order_mean = pd.merge(order_mean, order_num, on='user_id', how='left')
    order_mean = pd.merge(order_mean, order_last, on='user_id', how='left')
    order_mean['pre_time'] = order_mean['o_date'] + (order_mean['mean_time'] / order_mean['o_sku_num_x'])
    order_mean['pre_time_num'] = order_mean['o_date'] + (order_mean['mean_time'] / order_mean['o_sku_num_x'])*order_mean['o_sku_num_y']
    # print(order_mean)
    order_mean = order_mean[['user_id', 'pre_time_num']]
    order_mean = order_mean[order_mean['pre_time_num']>=label_time]
    order_mean[name] = 1
    order_mean = order_mean[['user_id', name]]
    df = df.merge(order_mean, on='user_id', how="left").fillna(0)
    return df
    # print(order_mean)


