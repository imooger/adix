import pandas as pd
import hashlib

from ..wizrenderer import WizRenderer
from ..dtype import *
from ..configs import *

from .calc import calc
from .wiz import wiz
 


def eda(df, col1=None, col2=None, vars=None, cache=None, wrap=True,stats=False):
    # Configuration settings
    cfg = Configs.current_theme

    # Get dtypes
    dtypes = Configs.get_dtypes(df)

    # Check if caching is enabled
    use_cache = Configs.use_eda_cache if cache is None else cache

    # Measure the start time
    #start_time = time.time()

    # Assuming eda_cache is the dictionary you want to get the size of
    memory_size = sys.getsizeof(Configs.eda_cache)
    
    # logger.debug(f"State of cache: {Configs.use_eda_cache}")
    # logger.debug(f"Memory size of eda_cache: {memory_size} bytes")
    # logger.debug(f"Number of cache keys: {(len(Configs.eda_cache.keys()))}")

    # Generate a hash key for caching based on function parameters
    cache_key = hashlib.sha256(f"{col1}-{col2}-{dtypes}".encode()).hexdigest()

    # Check if result is cached
    if use_cache and cache_key in Configs.eda_cache:
        cached_result = Configs.eda_cache[cache_key]
        cached_df = cached_result['df']

        # Now, compare the two DataFrames
        if col1 is None:
            are_equal = df.equals(cached_df)
        elif col1 == 'WRAPPER':
            are_equal = df.equals(cached_df)
        else:
            are_equal = df[col1].equals(cached_df[col1])
        if are_equal:
            # Return the cached result
            #logger.debug("Returning cached result.")
            #elapsed_time = time.time() - start_time
            #logger.debug(f"Execution time: {elapsed_time} seconds")
            return WizRenderer(cached_result['data_load'], cached_result['variable_type'], Configs.current_theme)

    # Calculate
    hub = calc(df, col1, col2, cfg, dtypes, vars, wrap)
    #print(hub)
    if hub is None:  # because plot(df) is recursive
        #logger.warning("Calculation result is None. its in init.py eda #calculate ")
        #elapsed_time = time.time() - start_time
        #logger.debug(f"Execution time: {elapsed_time} seconds")
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
        #logger.debug("Result saved to cache.")

    # Measure the end time
    #elapsed_time = time.time() - start_time
    #logger.debug(f"Execution time: {elapsed_time} seconds")

    # # STATS
    # print('stat',stats==True)
    # if stats == True:
    #     return hub
        
    # else:
    #     return WizRenderer(data_load, hub.variable_type, cfg)

    return WizRenderer(data_load, hub.variable_type, cfg)