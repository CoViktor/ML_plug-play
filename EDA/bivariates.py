import plotly as py
import plotly.express as px




# get imput on what target variable is
# -> do bivariates for each var & target
# plot numericals & categoricals in their own way
# -> look for the covar prints of each numerical var with target

# ALSO DEPENDS ON TYPE OF TARGET

# Bivariates correlation matrix
# first OHE the categorical columns

def plot_target_bivariates(df, target, type, columns=None):
    pass


import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import chisquare
import numpy as np


def num_y_bivariates(df, y, columns):
    for col in columns:
        if col in df.select_dtypes(include=['int64', 'float64']).columns:
            fig = px.scatter(df, x=col, y=y, title=f'{col} vs {y}')
            fig.show()
        elif col in df.select_dtypes(include=['object', 'category']).columns:
            fig = px.bar(df, x=col, y=y, title=f'{col} vs {y}')
            fig.show()


def cat_y_bivariates(df, y, columns):
    for col in columns:
        if col in df.select_dtypes(include=['int64', 'float64']).columns:
            fig = px.scatter(df, x=col, y=y, title=f'{col} vs {y}')
            fig.show()
        elif col in df.select_dtypes(include=['object', 'category']).columns:
            if df[y].dtype in ['object', 'category']:
                fig = px.bar(df, x=col, y=y, color=y, title=f'{col} vs {y}')
            else:
                fig = px.box(df, x=col, y=y, color=y, title=f'{col} vs {y}')
            fig.show()


def explore_bivariates(df, target, columns=None):
    if target in df.select_dtypes(include=['int64', 'float64']).columns:
        cat_y_bivariates(df, target)
    elif target in df.select_dtypes(include=['object', 'category']).columns:
        cat_y_bivariates(df, target)