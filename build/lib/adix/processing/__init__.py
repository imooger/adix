import pandas as pd
import hashlib

from ..wizrenderer import WizRenderer
from ..dtype import *
from ..configs import *

from .calc import calc
from .wiz import wiz
 


def eda(df, col1=None, col2=None, vars=None, cache=None, dash=True,stats=False,corr=False):
    
    # Disable cache if dash == 'only'
    if dash == 'only': cache = False

    # Disable cache if dash == 'only'
    if corr: cache = not corr

    
    # Configuration settings
    cfg = Configs.current_theme

    # Get dtypes
    dtypes = Configs.get_dtypes(df)

    # Check if caching is enabled
    use_cache = Configs.use_eda_cache if cache is None else cache

    

    # get the size of eda_cache dictionary
    memory_size = sys.getsizeof(Configs.eda_cache)
    
    
    # Generate a hash key for caching based on function parameters
    cache_key = hashlib.sha256(f"{col1}-{col2}-{dtypes}".encode()).hexdigest()

    # Check if result is cached
    if use_cache and cache_key in Configs.eda_cache:
        cached_result = Configs.eda_cache[cache_key]
        cached_df = cached_result['df']

        # compare the two DataFrames
        if col1 is None:
            are_equal = df.equals(cached_df)
        elif col1 == 'WRAPPER':
            are_equal = df.equals(cached_df)
        else:
            are_equal = df[col1].equals(cached_df[col1])
        if are_equal:
            # Return the cached result
            return WizRenderer(cached_result['data_load'], cached_result['variable_type'], Configs.current_theme)

    # Calculate
    hub = calc(df, col1, col2, cfg, dtypes, vars, dash,stats,corr)
    #print(hub)
    # hub.save()
    if hub is None:  # because plot(df) is recursive
        return 0

    # Render
    data_load = wiz(hub, cfg)
    
    # Save the result to cache
    if use_cache:
        Configs.eda_cache[cache_key] = {
            'data_load': data_load,
            'variable_type': hub.variable_type,
            'df': df.copy()  # Cache a copy of the DataFrame
        }


    return WizRenderer(data_load, hub.variable_type, cfg)
