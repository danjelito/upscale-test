import re
import pandas as pd

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