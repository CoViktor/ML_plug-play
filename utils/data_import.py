import pandas as pd
import json


def load_data(filepath, type):
    """
    Load data from a file.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data as a pandas DataFrame.
    """
    if type == 'csv':
        df = pd.read_csv(filepath)
    elif type == 'json':
        with open(filepath) as f:
            df = json.load(f)
    elif type == 'excel':
        df = pd.read_excel(filepath)
    elif type == 'pkl':
        df = pd.read_pickle(filepath)
    elif type == 'csv_with_delimiter':
        df = pd.read_csv(filepath, delimiter=',')
    elif type == 'csv_with_semicolon':
        df = pd.read_csv(filepath, delimiter=';')
    elif type == 'parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError('Invalid file type')
    return df