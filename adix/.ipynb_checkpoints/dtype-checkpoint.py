import numpy as np
import pandas as pd



def determine_variable_type(data_series, threshold_unique=0.05, threshold_missing=0.2, threshold_text_ratio=0.1):
    """
    Determine if an independent variable is categorical, continuous, or text based on the data.

    Parameters:
    - data_series: A pandas Series representing the independent variable.
    - threshold_unique: A threshold for unique values (default is 0.05).
    - threshold_missing: A threshold for missing values (default is 0.2).
    - threshold_text_ratio: A threshold for the ratio of text values (default is 0.1).

    Returns:
    - Tuple containing the variable type ('categorical', 'continuous', 'text', 'datetime', or 'unknown') and the number of unique values.
    """
    
    if isinstance(data_series, np.ndarray):
        return 'unknown', None

    num_unique = data_series.nunique()
    unique_values_ratio = num_unique / len(data_series)

    if pd.api.types.is_numeric_dtype(data_series):
        missing_values_ratio = data_series.isnull().mean()

        if unique_values_ratio <= threshold_unique or missing_values_ratio >= threshold_missing:
            return 'categorical', num_unique, unique_values_ratio
        else:
            return 'continuous', num_unique, unique_values_ratio
            
    elif pd.api.types.is_object_dtype(data_series):    
        if unique_values_ratio >= threshold_text_ratio:
            return 'text', num_unique, unique_values_ratio
        else:
            return 'categorical', num_unique, unique_values_ratio
            
    elif pd.api.types.is_datetime64_any_dtype(data_series):
        return 'datetime', num_unique, unique_values_ratio

    else:
        return 'unknown', data_series  # Handle other data types if needed
