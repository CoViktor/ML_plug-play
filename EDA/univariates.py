import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def explore_df(df):
    """
    Print the head, columns, count of null values,
    and count of unique values for each categorical column in the DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to explore.

    Returns:
        None
    """
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
    for column in df.columns:
        if column in ['Price', 'ConstructionYear','BedroomCount', 'LivingArea', 'TerraceArea', 'GardenArea', 'Facades', 'EnergyConsumptionPerSqm']:
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



def plot_numerical_data(df):
    """
    Plot interactive histograms and boxplots for numerical columns in the DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing numerical data to be plotted.

    Returns:
        None
    """
    # Filter only numerical columns
    numerical_df = df.select_dtypes(include=['float64', 'int64'])

    num_variables = len(numerical_df.columns)
    # Initialize the figure with a grid of subplots
    titles = [
        f"{col}<br><sup>Skewness: {numerical_df[col].skew():.2f}</sup>" for col in numerical_df.columns
        for _ in range(2)  # Repeat for histogram and boxplot
        ]

    fig = make_subplots(rows=1, cols=num_variables*2, subplot_titles=titles)

    for i, column in enumerate(numerical_df.columns):
        # Add histogram in the odd columns
        fig.add_trace(go.Histogram(x=numerical_df[column], name=f'{column} Histogram',
                                   marker_color='skyblue', opacity=0.75, nbinsx=20),
                      row=1, col=2*i+1)
        
        # Add boxplot in the even columns
        fig.add_trace(go.Box(y=numerical_df[column], name=f'{column} Boxplot',
                             marker_color='blue'),
                      row=1, col=2*i+2)

    # Update layout for better spacing and to show legend
    fig.update_layout(height=600, width=400 * num_variables,  # Adjust width for better fit
                      title_text="Interactive Histograms and Boxplots", showlegend=True)
    fig.show()


def plot_categorical_data(df):
    """
    Plot interactive count plots for categorical columns in the DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing categorical data to be plotted.

    Returns:
        None
    """
    # Filter only categorical columns
    categorical_df = df.select_dtypes(include=['object', 'category'])

    for column in categorical_df.columns:
        # Calculate value counts and reset index with clear column names
        counts = categorical_df[column].value_counts().reset_index()
        counts.columns = ['Category', 'Count']  # Rename columns for clarity

        # Generate the bar plot
        fig = px.bar(counts, x='Category', y='Count',
                     labels={'Category': column, 'Count': 'Count'},
                     title=f"Interactive Count Plot for {column}")
        fig.show()


def plot_distributions(df):
    """
    Plot distributions of both numerical and categorical data in the DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to plot distributions for.

    Returns:
        None
    """
    plot_numerical_data(df)
    plot_categorical_data(df)