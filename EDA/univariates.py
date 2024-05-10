import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def explore_df(df, columns=None):
    """
    Print the head, columns, count of null values,
    and count of unique values for each categorical column in the DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to explore.

    Returns:
        None
    """
    if colums is None:
        columns= df.columns
    print('first lines:')
    print(df.head())
    print('columns:')
    print(df.columns)
    print('count of null values:')
    print(df.isnull().sum())
    print('count of unique categorical values:')
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}:")
        print(df[col].value_counts())
    for column in df.select_dtypes(exclude=['object']).columns:
        print(f'--{column}--')
        # setting IQR
        df[column].dropna()
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        # identify outliers
        threshold = 1.5
        outliers = df[(df[column] < Q1 - threshold * IQR) | (df[column] > Q3 + threshold * IQR)]
        lower = df[ df[column] < Q1 - threshold * IQR)]
        upper = df[ df[column] > Q3 + threshold * IQR)]
        print(len(outliers), f'outliers \nLower: {len(lower)} ( {lower.max()} - {lower.min()} )\nUpper: {len(upper)} ( {upper.min()} - {upper.max()} )')



import plotly.express as px

def plot_numerical_data(data, columns):
    fig = px.layout.Grid(rows=1, columns=16, width=500, height=400)

    for column in columns:
        if df[column].dtype in ['int64', 'float64']:
            hist_data = px.data.gapminder().query("year == 2007")
            fig.add_trace(
                px.histogram(hist_data, x=column, color=hist_data.continent, barmode='overlay', title=f"{column}: Skewness: {data[column].skew():.2f}"),
                row=1, col=(i % 16) + 1
            )

    fig.update_layout(
        title_text="Interactive Histograms with Skewness Information",
        height=400 * 16,
        width=500 * 16,
        grid_gap=0
    )

    fig.show()


def plot_categorical_data(df, columns):
    # Filter only categorical columns
    for column in columns:
        if df[column].dtype in ['object', 'category']:
            # Calculate value counts and reset index with clear column names
            counts = df[column].value_counts().reset_index()
            counts.columns = ['Category', 'Count']

        # Generate the bar plot
        fig = px.bar(counts, x='Category', y='Count',
                     labels={'Category': column, 'Count': 'Count'},
                     title=f"Interactive Count Plot for {column}")
        fig.show()


def plot_distributions(df, columns):
    """
    Plot distributions of both numerical and categorical data in the DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to plot distributions for.

    Returns:
        None
    """
    plot_numerical_data(df, columns)
    plot_categorical_data(df, columns)