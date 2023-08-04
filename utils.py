import re
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
from math import isnan

def to_snake_case(name):
    """Convert string from camelcase to snake case."""

    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


def clean_df(df):
    """Clean dataframe to intended format."""

    unused_cols= ['id', 'response', 'accepted_cmp1', 'accepted_cmp2', 'accepted_cmp3', 
                'accepted_cmp4', 'accepted_cmp5', 'complain', 
                'z_cost_contact', 'z_revenue']
    
    age_bin_edges = [0, 10, 20, 30, 40, 50, 60, 150]  # Customize bin edges as per your requirement
    age_bin_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']

    return (df
        .rename(columns= lambda c: to_snake_case(c))
        .drop(columns= unused_cols)
        .assign(
            dt_customer= lambda df_: pd.to_datetime(df_['dt_customer'], dayfirst= True), 
            enroll_year= lambda df_: df_['dt_customer'].dt.year, 
            age= lambda df_: max(df_['dt_customer'].dt.year) - df_['year_birth'], 
            age_basket= lambda df_: pd.cut(df_['age'], bins= age_bin_edges, labels= age_bin_labels, right=False)
        )
        .drop(columns= ['dt_customer', 'year_birth', 'age'])
        .astype({
            'education': 'string',
            'marital_status': 'string',
            'income': 'float',
            'kidhome': 'float',
            'teenhome': 'float',
            'recency': 'float',
            'mnt_wines': 'float',
            'mnt_fruits': 'float',
            'mnt_meat_products': 'float',
            'mnt_fish_products': 'float',
            'mnt_sweet_products': 'float',
            'mnt_gold_prods': 'float',
            'num_deals_purchases': 'float',
            'num_web_purchases': 'float',
            'num_catalog_purchases': 'float',
            'num_store_purchases': 'float',
            'num_web_visits_month': 'float',
            'enroll_year': 'int',
            'age_basket': 'str',
        })
    )


def drop_outliers(df: pd.DataFrame, columns: list, method: str):
    """Drop outliers from a dataframe

    Args:
        df (pd.DataFrame): dataframe to drop outliers from
        columns (list): list of columns to check
        method (str): methid to use

    Returns:
        pd.DataFrame: dataframe with outliers dropped
    """

    index_outliers= []

    if method == 'z_score':
        for col in columns:
            lower_limit = df[col].mean() - 3*df[col].std()
            upper_limit = df[col].mean() + 3*df[col].std()
            indexes= df.loc[
                (df[col] < lower_limit) | (df[col] > upper_limit)
            ].index.to_list()
            index_outliers.extend(indexes)
    elif method == 'iqr':
        for col in columns:
            percentile05 = df[col].quantile(0.05)
            percentile95 = df[col].quantile(0.95)
            iqr = percentile95 - percentile05
            lower_limit = percentile05 - 1.5 * iqr
            upper_limit = percentile95 + 1.5 * iqr
            indexes= df.loc[
                (df[col] < lower_limit) | (df[col] > upper_limit)
            ].index.to_list()
            index_outliers.extend(indexes)
    else:
        print('method unrecognized')

    df= df.drop(index= index_outliers)
    return df


def hopkins_test(X: pd.DataFrame):
    """Hopkins test to check if array contains meaningful cluster
    Close to 1: the data is highly clustered, 
    0.5: random data, 
    Close to 0: uniformly distributed data
    
    Args:
        X (pd.DataFrame): df to test. Scale and get dummies first

    Returns:
        str: hopkins test result
    """

    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = random.sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(np.random.uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print (ujd, wjd)
        H = 0

    if H <= 0.5:
        print(f'Hopkins statistic= {H: .3f}: no meaningful clusters')
    elif H > 0.5:
        print(f'Hopkins statistic= {H: .3f}: there are meaningful clusters')