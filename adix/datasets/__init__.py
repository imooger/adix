import os
from os.path import dirname, join
import pandas as pd


def load_dataset(name: str or None):
    """
    Load a dataset by name or list all available datasets.
    
    name = 'all' -> lists all available datasets
    name = 'titanic' -> import titanic dataset
    
    Parameters:
        name (str or None): The name of the dataset to import.
                            Use 'all' to list all available datasets.
    
    Returns:
        pandas.DataFrame or List[str]: The imported dataset as a pandas DataFrame
                                       if name is provided.
                                       List of available datasets
                                       if name is None or 'all'.
    
    Raises:
        ValueError: If the specified dataset name is not found.
    """
    if name == 'all':
        return _list_available_datasets()
    else:
        path = _get_dataset_path_by_name(name)
        return pd.read_csv(path, engine='python',on_bad_lines='skip')


def _get_dataset_path_by_name(name: str):
    dir_path = dirname(__file__)
    files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

    lowercase = name.lower()
    if lowercase.endswith(".csv"):
        lowercase = os.path.splitext(lowercase)[0]

    if lowercase not in [os.path.splitext(f)[0] for f in files]:
        raise ValueError(f"Dataset {name} is not in the file.")

    return join(dir_path, f"{lowercase}.csv")


def _list_available_datasets():
    dir_path = dirname(__file__)
    files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
    return [os.path.splitext(f)[0] for f in files]
