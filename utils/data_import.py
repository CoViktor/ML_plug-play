import pandas as pd
import json


def load_csv(filepath):
    """
    Load data from a CSV file.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    return df


def load_json(filepath):
    """
    Load data from a JSON file.

    Parameters:
        filepath (str): The path to the JSON file.

    Returns:
        dict or list: The loaded data as a dictionary or list.
    """
    with open(filepath) as f:
        data = json.load(f)
    return data


def load_excel(filepath):
    """
    Load data from an Excel file.

    Parameters:
        filepath (str): The path to the Excel file.

    Returns:
        pandas.DataFrame: The loaded data as a pandas DataFrame.
    """
    df = pd.read_excel(filepath)
    return df


def load_csv_with_delimiter(filepath, delimiter):
    """
    Load data from a CSV file with a specified delimiter.

    Parameters:
        filepath (str): The path to the CSV file.
        delimiter (str): The delimiter used in the CSV file.

    Returns:
        pandas.DataFrame: The loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(filepath, delimiter=delimiter)
    return df


def load_parquet(filepath):
    """
    Load data from a Parquet file.

    Parameters:
        filepath (str): The path to the Parquet file.

    Returns:
        pandas.DataFrame: The loaded data as a pandas DataFrame.
    """
    df = pd.read_parquet(filepath)
    return df