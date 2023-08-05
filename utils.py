import random
import re
from math import isnan

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def to_snake_case(name):
    """Convert string from camelcase to snake case."""

    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("__([A-Z])", r"_\1", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def clean_df(df):
    """Clean dataframe to intended format."""

    unused_cols = [
        "id",
        "response",
        "accepted_cmp1",
        "accepted_cmp2",
        "accepted_cmp3",
        "accepted_cmp4",
        "accepted_cmp5",
        "complain",
        "z_cost_contact",
        "z_revenue",
    ]

    string_cols = ["education", "marital_status", "age_basket"]

    float_cols = [
        "income",
        "recency",
        "mnt_wines",
        "mnt_fruits",
        "mnt_meat_products",
        "mnt_fish_products",
        "mnt_sweet_products",
        "mnt_gold_prods",
        "num_deals_purchases",
        "num_web_purchases",
        "num_catalog_purchases",
        "num_store_purchases",
        "num_web_visits_month",
    ]

    int_cols = ["enroll_year", "num_kids"]

    map_edu = {
        "Graduation": "University",
        "PhD": "Higher Education",
        "Master": "Higher Education",
        "Basic": "Basic Education",
        "2n Cycle": "Basic Education",
    }

    map_marital = {
        "Single": "No Partner",
        "Together": "Have Partner",
        "Married": "Have Partner",
        "Divorced": "No Partner",
        "Widow": "No Partner",
        "Alone": "No Partner",
        "Absurd": "Unknown",
        "YOLO": "Unknown",
    }

    # bin to group age
    age_bin_edges = [0, 10, 20, 30, 40, 50, 60, 150]
    age_bin_labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]

    return (
        df.rename(columns=lambda c: to_snake_case(c))
        .drop(columns=unused_cols)
        .assign(
            dt_customer=lambda df_: pd.to_datetime(df_["dt_customer"], dayfirst=True),
            enroll_year=lambda df_: df_["dt_customer"].dt.year,
            age=lambda df_: max(df_["dt_customer"].dt.year) - df_["year_birth"],
            age_basket=lambda df_: pd.cut(
                df_["age"], bins=age_bin_edges, labels=age_bin_labels, right=False, ordered= True
            ),
            education=lambda df_: df_["education"].map(map_edu),
            marital_status=lambda df_: df_["marital_status"].map(map_marital),
            num_kids=lambda df_: df_["kidhome"] + df_["teenhome"],
            is_parent=lambda df_: df_["num_kids"] > 0,
            total_spent=lambda df_: df_.loc[:, df_.columns.str.contains("mnt")].sum(
                axis=1
            ),
            total_purchase=lambda df_: df_.loc[
                :,
                df_.columns.str.contains(
                    "web_purchase|catalog_purchase|store_purchase", regex=True
                ),
            ].sum(axis=1),
        )
        .drop(columns=["dt_customer", "year_birth", "age", "kidhome", "teenhome"])
        .astype(
            {
                **{k: "string" for k in string_cols},
                **{k: "float" for k in float_cols},
                **{k: "int" for k in int_cols},
            }
        )
    )


def drop_outliers(df: pd.DataFrame, columns: list, method: str):
    """
    Drop outliers from a dataframe

    Args:
        df (pd.DataFrame): dataframe to drop outliers from
        columns (list): list of columns to check
        method (str): methid to use

    Returns:
        pd.DataFrame: dataframe with outliers dropped
    """

    index_outliers = []

    # remove samples that are further than 3 STD from mean
    if method == "z_score":
        for col in columns:
            lower_limit = df[col].mean() - 3 * df[col].std()
            upper_limit = df[col].mean() + 3 * df[col].std()
            indexes = df.loc[
                (df[col] < lower_limit) | (df[col] > upper_limit)
            ].index.to_list()
            index_outliers.extend(indexes)

    # remove bottom 5% and top 5% of samples
    elif method == "iqr":
        for col in columns:
            perc05 = df[col].quantile(0.05)
            perc95 = df[col].quantile(0.95)
            iqr = perc95 - perc05
            lower_limit = perc05 - 1.5 * iqr
            upper_limit = perc95 + 1.5 * iqr
            indexes = df.loc[
                (df[col] < lower_limit) | (df[col] > upper_limit)
            ].index.to_list()
            index_outliers.extend(indexes)

    else:
        print("method unrecognized")

    df = df.drop(index=index_outliers)
    return df


def hopkins_test(X: pd.DataFrame):
    """
    Hopkins test to check if array contains meaningful cluster
    Close to 1: the data is highly clustered,
    0.5: random data,
    Close to 0: uniformly distributed data

    Args:
        X (pd.DataFrame): df to test. Scale and get dummies first

    Returns:
        str: hopkins test result
    """

    n_features = X.shape[1]
    n_samples = X.shape[0]
    m = int(0.1 * n_samples)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = random.sample(range(0, n_samples, 1), m)

    ujd = []  # uniform random distance
    wjd = []  # within data distance
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(
            np.random.uniform(
                np.amin(X, axis=0), np.amax(X, axis=0), n_features
            ).reshape(1, -1),
            2,
            return_distance=True,
        )
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(
            X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True
        )
        wjd.append(w_dist[0][1])

    if not ujd or not wjd:
        return "Insufficient data for Hopkins test"

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0

    if H <= 0.5:
        return f"Hopkins statistic = {H: .3f}: no meaningful clusters"
    elif H > 0.5:
        return f"Hopkins statistic = {H: .3f}: there are meaningful clusters"


def catchstate(df, var_name: str) -> "pd.DataFrame":
    """
    Helper function that captures intermediate Dataframes mid-chain.
    In the global namespace, make a new variable called var_name and set it to dataframe
    """
    globals()[var_name] = df
    return df


def preprocess_df(df):
    scaler = StandardScaler()
    imputer = KNNImputer(n_neighbors=5, weights="uniform")

    return (
        df
        # one hot encoding
        .pipe(pd.get_dummies)
        # scaling
        .pipe(catchstate, "current_df")
        .assign(
            **{
                k: scaler.fit_transform(current_df[k].values.reshape(-1, 1)).flatten()
                for k in current_df.columns
            },
        )
        # imputing
        .pipe(catchstate, "current_df")
        .assign(
            **{
                k: imputer.fit_transform(current_df[k].values.reshape(-1, 1)).flatten()
                for k in current_df.columns
            },
        )
    )


def plot_violin(x: str, y: str, df: pd.DataFrame):
    """Violin plot

    Args:
        x (str): segment
        y (str): feature
        df (pd.DataFrame)
    """

    fig, ax = plt.subplots(figsize=(9, 3), dpi=150)

    sns.violinplot(x=df[x], y=df[y], ax=ax, palette="tab10", alpha=0.8)

    ax.set_title(f"{y.title()} by {x.title()}", fontweight="bold", fontsize=16)
    ax.set_xlabel(x.title(), fontweight="bold")
    ax.set_ylabel(y.title(), fontweight="bold")
    ax.spines[["right", "top"]].set_visible(False)
    plt.show()


def plot_bar_with_hue(hue: str, df: pd.DataFrame):
    """Bar plot per segment with hue

    Args:
        hue (str): feature to divide per segment
        df (pd.DataFrame): dataframe to process
    """

    x = "segment"
    y = "percentage"

    data = (
        df.groupby([x, hue])
        .agg(count=(hue, "size"))
        .assign(
            percentage=lambda df_: (
                df_["count"] / df_.groupby(x)["count"].transform("sum")
            ).round(3)
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 3), dpi=150)
    ax = sns.barplot(x=data[x], y=data[y], hue=data[hue], palette="Greens")
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".0%"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=7,
            color="black",
        )

    hue = hue.replace("_", " ")
    ax.set_title(f"{hue.title()} by {x.title()}", fontweight="bold", pad=30)
    ax.set_xlabel(x.title(), fontweight="bold")
    ax.set_ylabel(y.title(), fontweight="bold")
    ax.spines[["right", "top"]].set_visible(False)
    sns.move_legend(
        ax, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncols=6, title=None
    )

    ylim_max = ax.get_ylim()[1]
    ax.set_ylim(0, ylim_max * 1.1)

    ax.grid(axis="y", linestyle="--")

    plt.show()


def plot_bar(x, y, df, ylim=None):
    """Bar plot per segment

    Args:
        x (str): x-axis
        y (str): y-axis
        df (pd.DataFrame)
        ylim (tuple, optional): y-lim if any. Defaults to None.
    """

    fig, ax = plt.subplots(figsize=(9, 3), dpi=150)

    sns.barplot(x=df[x], y=df[y], palette="Greens", ax=ax)
    ax.set_title(f'{y.title().replace("_", " ")} by {x.title()}', fontweight="bold")
    ax.set_xlabel(x.title().replace("_", " "), fontweight="bold")
    ax.set_ylabel(y.title().replace("_", " "), fontweight="bold")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(ylim)
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".1f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )
    plt.show()


def plot_heatmap(df: pd.DataFrame, title: str):
    """Plot heatmap of percentage with annotation

    Args:
        df (pd.DataFrame)
        title (str): plot title
    """

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    sns.heatmap(df, cmap="Greens", annot=True, fmt=".1%", cbar=False, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Segment", fontweight="bold")
    xlabels = [
        x.get_text().replace("mnt", "").replace("_", " ").split(" ")[1].strip().title()
        for x in ax.get_xticklabels()
    ]
    ax.set_xticklabels(xlabels)

    plt.show()


def plot_scatter(x: str, y: str, df):
    """Scatter plot with centroids marked

    Args:
        x (str): x-axis
        y (str): y-axis
        df (_type_): dataframe with features
    """

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150, facecolor="white")

    sns.scatterplot(data=df, x=x, y=y, hue="segment", alpha=0.75, ax=ax)

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title(
        f'{y.title().replace("_", " ")} vs {x.title().replace("_", " ")}',
        fontweight="bold",
        pad=30,
    )
    ax.set_xlabel(x.title().replace("_", " "), fontweight="bold")
    ax.set_ylabel(y.title().replace("_", " "), fontweight="bold")
    sns.move_legend(ax, "upper center", title=None, ncol=5, bbox_to_anchor=(0.5, 1.1))
    ax.set_facecolor("white")
    plt.show()
