from .dtype import *

import sys



class Theme:

    PEACH =  {
    'dash_donuts_color': '#FFCBA5',
    'dash_bars_color': '#FF9899',
    'dash_bars_text_color': '#525252',
    'label_color': '#FF9899',
    'mini_hist_color': '#FFCBA5',
    'hist_color': '#FFCBA5',
    'hist_kde_color': '#FF9899',
    'bar_color': 'light:#FF9899',
    'bar_font_color': '#525252',
    'hover_color': '#FF9899'}

    FOREST = {
    'dash_donuts_color': '#a3b18a',
    'mini_hist_color': '#a3b18a',
    'hist_color': '#a3b18a',
    
    'label_color':'#bc6c25',
    'dash_bars_color': '#bc6c25',
    'hist_kde_color': '#bc6c25',
    'bar_color': 'light:#bc6c25',
    'hover_color': '#a3b18a',
    
    'dash_bars_text_color': '#525252',
    'bar_font_color': '#525252',}

    AMERICANA = {
    'dash_donuts_color': '#c1121f',
    'mini_hist_color': '#c1121f',
    'hist_color': '#c1121f',
    
    'label_color':'#457b9d',
    'dash_bars_color': '#457b9d',
    'hist_kde_color': '#457b9d',
    'bar_color': 'light:#457b9d',
    'hover_color': '#fdf0d5',
    
    'dash_bars_text_color': '#525252',
    'bar_font_color': '#525252',}

    BABY = {
    'dash_donuts_color': '#a2d2ff',
    'mini_hist_color': '#a2d2ff',
    'hist_color': '#a2d2ff',
    
    'label_color':'#ff8fab',
    'dash_bars_color': '#ff8fab',
    'hist_kde_color': '#ff8fab',
    'bar_color': 'light:#ff8fab',
    'hover_color': '#ff8fab',
    
    'dash_bars_text_color': '#525252',
    'bar_font_color': '#525252',}

    PASTEL = {
    'dash_donuts_color': '#e4c1f9',
    'mini_hist_color': '#d0f4de',
    'hist_color': '#a9def9',
    
    'label_color':'#a9def9',
    'dash_bars_color': '#ff99c8',
    'hist_kde_color': '#e4c1f9',
    'bar_color': 'light:#e4c1f9',
    'hover_color': '#fcf6bd',
    
    'dash_bars_text_color': '#525252',
    'bar_font_color': '#525252',}

    ITALY = {
    'dash_donuts_color': '#e5383b',
    'mini_hist_color': '#e5383b',
    'hist_color': '#f07167',
    
    'label_color':'#55a630',
    'dash_bars_color': '#55a630',
    'hist_kde_color': '#55a630',
    'bar_color': 'light:#55a630',
    'hover_color': '#55a630',
    
    'dash_bars_text_color': '#525252',
    'bar_font_color': '#525252',}

    OCEAN = {
    'dash_donuts_color': '#fed9b7',
    'mini_hist_color': '#fed9b7',
    'hist_color': '#fed9b7',
    
    'label_color':'#00afb9',
    'dash_bars_color': '#00afb9',
    'hist_kde_color': '#00afb9',
    'bar_color': 'light:#00afb9',
    'hover_color': '#00afb9',
    
    'dash_bars_text_color': '#525252',
    'bar_font_color': '#525252',}

    PLATINUM = {
    'dash_donuts_color': '#c6c5b9',
    'mini_hist_color': '#c6c5b9',
    'hist_color': '#c6c5b9',
    
    'label_color':'#393d3f',
    'dash_bars_color': '#393d3f',
    'hist_kde_color': '#393d3f',
    'bar_color': 'light:#393d3f',
    'hover_color': '#bcb8b1',
    
    'dash_bars_text_color': '#525252',
    'bar_font_color': '#525252',}

    GOLD = {
    'dash_donuts_color': '#ffc43d',
    'mini_hist_color': '#ffc43d',
    'hist_color': '#ffc43d',
    
    'label_color':'#f4d35e',
    'dash_bars_color': '#f4d35e',
    'hist_kde_color': '#0b132b',
    'bar_color': 'light:#f4d35e',
    'hover_color': '#f4d35e',
    
    'dash_bars_text_color': '#525252',
    'bar_font_color': '#525252',}
    

class Configs:

    # cache flag
    use_eda_cache = True
    
    
    # Simple cache to store previously generated plots
    eda_cache = {}

    @classmethod
    def reset_cache(cls, df):
        # Iterate over keys in the cache and remove entries that correspond to the given DataFrame
        keys_to_remove = [key for key in cls.eda_cache.keys() if cls.eda_cache[key]['df'].equals(df)]
        for key in keys_to_remove:
            del cls.eda_cache[key]

        print(f"""Cache reset for DataFrame with shape {df.shape}\n
                number of cache keyslen{(Configs.eda_cache.keys())}"""            
                     )





    ################################ setting DTYPES #####################################################

    
    dtypes_cache = {}
    dtype = None

    use_id = True
    
    @classmethod
    def get_dtypes(cls, df):
        """Returns a dict of suggested dtypes for each variable"""
        if cls.use_id:
            df_name = id(df)
        else:
            df_name = str(df)
        
        cls.dtype = cls.dtypes_cache.get(df_name,None)   
        
        if cls.dtype is not None:
            return cls.dtypes_cache[df_name]
        else:
            cls.dtypes_cache[df_name] = {i: determine_variable_type(df[i])[0] for i in df.columns}       
        return cls.dtypes_cache[df_name]


    @classmethod
    def set_dtypes(cls,df, dtypes_dict=None):
        """Updates the dtypes attribute with the provided dictionary"""
        if cls.use_id:
            df_name = id(df)
        else:
            df_name = str(df)

        if dtypes_dict == 'reset':
            cls.dtypes_cache[df_name] = None
               
        cls.dtype = cls.dtypes_cache.get(df_name,None)
        
        if cls.dtype is None or dtypes_dict is None:
            cls.dtype = cls.get_dtypes(df)  # Initialize dtypes using get_dtypes if it's None
        else:
            for key, new_value in dtypes_dict.items():
                if key in cls.dtype and cls.dtype[key] != new_value:
                    cls.dtype[key] = new_value
                    cls.dtypes_cache[df_name] = cls.dtype
                    #print(cls.dtype)

        return 'done'



    
    ################################ THEME #####################################################
    
    current_theme = Theme.PEACH

    @classmethod
    def get_theme(cls, current=False):
        """
        List available themes and their values.

        Parameters:
        - values (bool, optional): If True, returns a dictionary with theme names and values.
          If False, returns a list of theme names only.

        Returns:
        - dict or list: If values is True, a dictionary of theme names and their values.
          If values is False, a list of theme names.
          
        """
        if current:
            return cls.current_theme.copy()
            # themes = {}
            # for attr in cls.current_theme:
            #     if not callable(getattr(Theme, attr)) and not attr.startswith("__"):
            #         themes[attr] = getattr(Theme, attr)
            # return themes
        else:
            return [attr for attr in dir(Theme) if not callable(getattr(Theme, attr)) and not attr.startswith("__")]

    @classmethod
    def set_theme(cls, theme_name):
        """Set the current theme based on the provided theme name."""
        if hasattr(Theme, theme_name):
            cls.use_eda_cache = False
            cls.eda_cache = {}
            cls.current_theme = getattr(Theme, theme_name)
            cls.use_eda_cache = True
            
            
        else:
            print(f"Theme '{theme_name}' not found. Using the default theme.")

    @classmethod
    def add_theme(cls, theme_name, theme_values):
        """
        Add a new theme to the Theme class.
    
        Parameters:
        - cls: The Theme class.
        - theme_name (str): The name of the new theme.
        - theme_values (dict): A dictionary containing the values of the new theme.
    
        Returns:
        - None
    
        Example:
        >>> Configs.add_theme('NEW_THEME', {'hist_color': 'purple', 'tabset': '#FFA500'})
        """
        setattr(Theme, theme_name, theme_values)