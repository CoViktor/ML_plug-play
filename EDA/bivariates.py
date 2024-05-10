import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr


def num_y_bivariates(df, y, columns):
    """
    Generate and display bivariate plots for numerical features against a target variable.
    
    For each column in `columns`, if it is numeric, this function will create a scatter plot 
    against the target variable `y`. If the column is categorical, it will create a bar plot.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - y (str): The name of the target variable column.
    - columns (list): The list of column names to be plotted against the target variable.
    
    Returns:
    None: The function will display the plots and does not return anything.
    """
    for col in columns:
        # Check if the column is numeric
        if col in df.select_dtypes(include=['int64', 'float64']).columns:
            # Create and display a scatter plot for the numeric column against the target variable
            fig = px.scatter(df, x=col, y=y, title=f'{col} vs {y}')
            fig.show()
        # Check if the column is categorical
        elif col in df.select_dtypes(include=['object', 'category']).columns:
            # Create and display a bar plot for the categorical column against the target variable
            fig = px.bar(df, x=col, y=y, title=f'{col} vs {y}')
            fig.show()


def cat_y_bivariates(df, y, columns):
    """
    Generate and display bivariate plots for categorical features against a target variable.
    
    For each column in `columns`, if it is numeric, this function will create a scatter plot 
    against the target variable `y`. If the column is categorical, it will create either a bar 
    plot or a box plot depending on the type of the target variable `y`.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - y (str): The name of the target variable column.
    - columns (list): The list of column names to be plotted against the target variable.
    
    Returns:
    None: The function will display the plots and does not return anything.
    """
    for col in columns:
        # If column is numeric, create a scatter plot
        if col in df.select_dtypes(include=['int64', 'float64']).columns:
            fig = px.scatter(df, x=col, y=y, title=f'{col} vs {y}')
            fig.show()
        # If column is categorical, decide which plot to create based on the target variable type
        elif col in df.select_dtypes(include=['object', 'category']).columns:
            # If target variable is categorical, create a bar plot
            if df[y].dtype in ['object', 'category']:
                fig = px.bar(df, x=col, y=y, color=y, title=f'{col} vs {y}')
            # If target variable is numerical, create a box plot
            else:
                fig = px.box(df, x=col, y=y, color=y, title=f'{col} vs {y}')
            fig.show()


def correlations(df, columns):
    """
    Calculate and display the correlation matrix for a given DataFrame and selected columns.
    
    This function first encodes categorical variables using One-Hot Encoding and then calculates
    the correlation matrix for the DataFrame. It plots a heatmap to visualize the correlations
    and prints out any significant correlations with the target variable as well as strong
    correlations between predictor variables.
    
    Parameters:
    - df (pandas.DataFrame): The input DataFrame.
    - columns (list of str): The list of columns to include in the correlation analysis.
    
    Outputs:
    - A heatmap plot of the correlation matrix.
    - Printed statements of significant and strong correlations.
    """
    
    # Determine categorical columns and perform One-Hot Encoding
    categorical_columns = df[columns].select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_columns]).toarray()
    feature_names = encoder.get_feature_names_out()
    
    # Drop original categorical columns and concatenate encoded data
    df_encoded = df.drop(columns=categorical_columns)
    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoded_data, columns=feature_names, index=df.index)], axis=1)
    
    # Calculate the correlation matrix for the encoded DataFrame
    correlation_matrix = df_encoded.corr()
    
    # Plot the heatmap for the correlation matrix
    fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        annotation_text=correlation_matrix.round(2).values,
        showscale=True,
        colorscale='Viridis'
    )
    fig.show()
    
    # Identify and print significant correlations with the target variable
    print('\nSignificant correlations with target:')
    for feature_name in df_encoded.columns:
        if pd.api.types.is_numeric_dtype(df_encoded[feature_name]):
            valid_idx = df_encoded[feature_name].notna()
            for target_column in df_encoded.columns:
                if target_column != feature_name:
                    # Compute the Pearson correlation coefficient
                    corr, _ = pearsonr(df_encoded.loc[valid_idx, feature_name], df_encoded.loc[valid_idx, target_column])
                    # Report significant correlations (absolute value of correlation >= 0.5)
                    if abs(corr) >= 0.5:
                        print(f"Significant correlation between {feature_name} and {target_column}: {round(corr, 2)}")
    
    # Identify and print strong correlations between predictor variables
    print('\nStrong correlations between predictor variables:')
    for i in range(len(df_encoded.columns)):
        for j in range(i+1, len(df_encoded.columns)):
            feature_name1 = df_encoded.columns[i]
            feature_name2 = df_encoded.columns[j]
            if pd.api.types.is_numeric_dtype(df_encoded[feature_name1]) and pd.api.types.is_numeric_dtype(df_encoded[feature_name2]):
                valid_idx = df_encoded[feature_name1].notna() & df_encoded[feature_name2].notna()
                # Compute the Pearson correlation coefficient
                corr, _ = pearsonr(df_encoded.loc[valid_idx, feature_name1], df_encoded.loc[valid_idx, feature_name2])
                # Report strong correlations (absolute value of correlation > 0.7)
                if abs(corr) > 0.7:
                    print(f"Strong correlation between {feature_name1} and {feature_name2}: {round(corr, 2)}")


def explore_bivariates(df, target, columns):
    """
    Explore bivariate relationships between the target variable and each feature in columns.

    Depending on the data type of the target variable, it calls the appropriate function
    to generate and display the plots. Additionally, it calculates and displays the 
    correlation between features if the target variable is numeric.

    Parameters:
    df (DataFrame): The dataframe containing the dataset.
    target (str): The name of the target variable.
    columns (list): List of column names to explore in relation to the target variable.
    """

    # Check if the target variable is numeric and call cat_y_bivariates accordingly
    if target in df.select_dtypes(include=['int64', 'float64']).columns:
        # Numeric target: Explore bivariate relationships with cat_y_bivariates
        cat_y_bivariates(df, target, columns)
    # Check if the target variable is categorical and call cat_y_bivariates accordingly
    elif target in df.select_dtypes(include=['object', 'category']).columns:
        # Categorical target: Explore bivariate relationships with cat_y_bivariates
        cat_y_bivariates(df, target, columns)
    
    # Calculate and display correlations for numeric columns
    correlations(df, columns)
