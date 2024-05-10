import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr


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



def correlation_matrix_heatmap(df, columns):
    # Determine categorical columns & OHE
    categorical_columns = df[columns].select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_columns]).toarray()
    feature_names = encoder.get_feature_names_out()
    df_encoded = df.drop(columns=categorical_columns)
    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoded_data, columns=feature_names, index=df.index)], axis=1)
    
    # Calculate the correlation matrix
    correlation_matrix = df_encoded.corr()
    
    # Plot the heatmap
    fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        annotation_text=correlation_matrix.round(2).values,
        showscale=True,
        colorscale='Viridis'
    )
    fig.show()
    
    # Print significant correlations
    for feature_name in df_encoded.columns:
        if pd.api.types.is_numeric_dtype(df_encoded[feature_name]):
            valid_idx = df_encoded[feature_name].notna()
            for target_column in df_encoded.columns:
                if target_column != feature_name:
                    corr, _ = pearsonr(df_encoded.loc[valid_idx, feature_name], df_encoded.loc[valid_idx, target_column])
                    if abs(corr) >= 0.5:
                        print(f"Significant correlation between {feature_name} and {target_column}: {round(corr, 2)}")


def explore_bivariates(df, target, columns=None):
    if target in df.select_dtypes(include=['int64', 'float64']).columns:
        cat_y_bivariates(df, target)
    elif target in df.select_dtypes(include=['object', 'category']).columns:
        cat_y_bivariates(df, target)