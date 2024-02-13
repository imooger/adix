from ...dtype import *
from ...datahub import *
from ...configs import *



import pandas as pd
import numpy





def calc(df=None, x=None, y=None, cfg=None, dtypes=None, vars=None, wrap=True):

    """Call function based on the number of arguments (all, uni, bi)"""
    
    if df is not None and wrap=='only':
        return calculate_wrapper(df, cfg, dtypes)
    
    elif df is not None and x is None and y is None:
        explore_univariate(df, dtypes, vars, wrap)
       
    elif df is not None and x == 'WRAPPER':
        return calculate_wrapper(df, cfg, dtypes)
    
    elif x is not None and y is None:
        return calculate_univariate(df, x, cfg, dtypes)
    elif x is not None and y is not None:
        return calculate_bivariate(df, x, y, cfg, dtypes)
    else:
        raise ValueError("Invalid number of arguments")

def calculate_wrapper(df, cfg, dtypes):

    n_obs, n_var = df.shape
    
    nas = df.isna().sum().sum()
    dup = df.duplicated().sum()
    
    dtx = Configs.get_dtypes(df)

    unique_counts = df.nunique()
    
    # Percentage of missing values
    missing_percentage = (df.isnull().mean().mean() * 100).round(1)
    
    # Percentage of unique rows
    unique_rows_percentage = (df.duplicated().sum() / df.shape[0] * 100).round(1)
    
    # Percentage of constant columns
    constant_columns_percentage = ((df.nunique() == 1).mean() * 100).round(1)
    
    wrapper_data = {
        'Number of columns': n_var,
        'Number of rows':	n_obs,
        'Missing cells':	nas,
        'Missing cells (%)':	missing_percentage,
        'Duplicate rows':	dup,
        'Duplicate rows (%)':	unique_rows_percentage,
        'Constant columns (%)' : constant_columns_percentage,
        'Total memory size in KB':	np.round(df.memory_usage().sum() / 1024,2),
        'Avg. record size in Bytes':	np.round(df.memory_usage().sum() / n_obs,2),

        #'Variable types'
    
        'Numeric':	sum(value == 'continuous' for value in dtx.values()),
        'Categorical':	sum(value == 'categorical' for value in dtx.values()),
        'Text':	sum(value == 'text' for value in dtx.values()),
        'Datetime': sum(value == 'datetime' for value in dtx.values()),

        ## WITH %%  find alert and add % at the end of -> !important;'>{inner_value[2]} %</td></tr>"
        
        # 'missing': {col: [missing_count := df[col].isnull().sum(),'values', np.round(missing_count / n_obs * 100,1)] for col in df.columns if df[col].isnull().any()},
        # 'unique': {col: [unique_counts[col],'values','U'] for col in df.columns if unique_counts[col] == df.shape[0]},
        # 'constant': {col: [unique_counts[col],'value','C'] for col in df.columns if unique_counts[col] == 1},
        # 'zero': {col: [(df[col] == 0).sum(),'values','Z'] for col in df.columns if (df[col] == 0).any()},
        # 'negative': {col: [(df[col] < 0).sum(),'values','N'] for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and (df[col] < 0).any()}

        ## WITHOUT %
        'missing': {col: [missing_count := df[col].isnull().sum(),'values',] for col in df.columns if df[col].isnull().any()},
        'unique': {col: [unique_counts[col],'values',] for col in df.columns if unique_counts[col] == df.shape[0]},
        'constant': {col: [unique_counts[col],'value',] for col in df.columns if unique_counts[col] == 1},
        'zero': {col: [(df[col] == 0).sum(),'values',] for col in df.columns if (df[col] == 0).any()},
        'negative': {col: [(df[col] < 0).sum(),'values',] for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and (df[col] < 0).any()}
    }
    #print(wrapper_data)    
    return DataHub(col=df, data=wrapper_data, variable_type='wrapper')
    

def explore_univariate(df, dtypes, vars=None, wrap=True):
    """Explore univariate analysis for all columns"""
    from .. import eda
            
    cal_columns = df.columns.insert(0,'WRAPPER') if wrap else df.columns
        
    if vars:
        dtypes = Configs.get_dtypes(df)
        selected_vars = [var for var, dtype in dtypes.items() if dtype == vars]
        
        # print('dtypes',dtypes)
        # print('sel.vars',selected_vars)

        # plot_instance = eda(df,wrap=False)
        # plot_instance.show()   
        for column in selected_vars:
            plot_instance = eda(df, column)
            plot_instance.show()
        
    else:
        # plot_instance = eda(df,wrap=False)
        # plot_instance.show()
        for column in cal_columns:
            plot_instance = eda(df, column)
            plot_instance.show()
            


def calculate_univariate(df, x, cfg, dtypes):
    """Calculate univariate analysis for a single variable"""
    col = df[x]
    dtype = get_dtype_info(col,x, dtypes)
    data = cont_comps(col, cfg, dtype)
    
    if dtype[0] == 'categorical' and data['MISSING'] > 0:
        col = col.dropna()
    
    return DataHub(col=col, data=data, variable_type=dtype[0])


def calculate_bivariate(df, x, y, cfg, dtypes):
    """Calculate bivariate analysis for two variables"""
    dtype1 = get_dtype_info(df[x],x, dtypes)
    dtype2 = get_dtype_info(df[y],y, dtypes)

    col1 = df[x]
    col2 = df[y]
    
    #print(dtype1, dtype2)

    if dtype1[0] == 'continuous' and dtype2[0] == 'continuous':
        return DataHub(col1=col1, col2=col2, data=None, variable_type='biv_con')
    elif (dtype1[0] == 'continuous' and dtype2[0] == 'categorical') or (dtype2[0] == 'continuous' and dtype1[0] == 'categorical'):
        return DataHub(col1=col1, col2=col2, data=(dtype1, dtype2), variable_type='biv_con_cat')
    elif dtype1[0] == 'categorical' and dtype2[0] == 'categorical':
        return DataHub(col1=col1, col2=col2, data=None, variable_type='biv_cat')
    else:
        print('compute_bivariate -> biv in progress')
        return None


def get_dtype_info(col,var_name, dtypes):
    """Get dtype information for a column"""
    if dtypes is None:
        return determine_variable_type(col)
    else:
        dtype_info = dtypes.get(var_name, determine_variable_type(col)[0])
        col_un = col.nunique()
        return dtype_info, col_un, col_un / len(col)



def cont_comps(ser = None, cfg = None, dtype = None):
    # ser = pd.series -> df.column => series
    
    def series_memory_usage(col, deep=False):
        total_memory_bytes = col.memory_usage(deep=deep)

        if total_memory_bytes < 1_000_000:
            total_memory_mb = total_memory_bytes / (1024)
            return f"{total_memory_mb:.2f} KB"
        else:
            total_memory_mb = total_memory_bytes / (1024 ** 2)
            return f"{total_memory_mb:.2f} MB"

    dt = dtype[0]
    #print(dtype)
    if dt == 'unknown':
        return ser
    if isinstance(ser, np.ndarray):
        return ser
    else:
        data = {
            'TOTAL': ser.size,
            'VALUES': ser.count(),
            'VALUES_P': np.round((ser.count() / ser.size) * 100, 2),
            'MISSING': ser.isna().sum(),
            'MISSING_P': np.round((ser.isna().sum() / ser.size) * 100, 2),
            'DISTINCT': dtype[1],
            'DISTINCT_P': np.round(dtype[2] * 100, 2),
            '': '',
            'MEMORY': series_memory_usage(ser),
            'DTYPE': ser.dtype,
            'v_type': dt
        }
    
    
    if dt == 'categorical':
        #print('categorical is comming next')
        None
    elif dt == 'continuous':
        data.update({
            'MAX': np.round(np.nanmax(ser), 2),
            '95%': np.round(np.nanpercentile(ser, 95), 2),
            'Q3': np.round(np.nanpercentile(ser, 75), 2),
            'AVG': np.round(np.nanmean(ser), 1),
            'MEDIAN': np.round(np.nanmedian(ser), 1),
            'Q1': np.round(np.nanpercentile(ser, 25), 1),
            '5%': np.round(np.nanpercentile(ser, 5), 2),
            'MIN': np.round(np.nanmin(ser), 2),
            'RANGE': np.round(np.nanmax(ser) - np.nanmin(ser), 1),
            'IQR': np.round(np.nanpercentile(ser, 75) - np.nanpercentile(ser, 25), 1),
            'STD': np.round(np.nanstd(ser), 1),
            'VAR': np.round(np.nanvar(ser)),
            ' ': ' ',
            'KURT.': np.round(ser.kurtosis(), 3),
            'SKEW': np.round(ser.skew(), 3),
            'SUM': np.round(np.nansum(ser), 3)
        })

    elif dt == 'text': 
        w_len = ser.str.len()
        data.update({
        'w_len' : w_len,
        'Max length' : w_len.max(),
        'Mean length' : np.round(w_len.mean(),2),
        'Median length' : np.round(w_len.median(),2),
        'Min length' : w_len.min()
        })

    elif dt == 'datetime':
        data.update({
            'MAX': ser.max(),
            # '95%': np.round(np.nanpercentile(ser, 95), 2),
            # 'Q3': np.round(np.nanpercentile(ser, 75), 2),
            # 'AVG': np.round(np.nanmean(ser), 1),
            # 'MEDIAN': np.round(np.nanmedian(ser), 1),
            # 'Q1': np.round(np.nanpercentile(ser, 25), 1),
            # '5%': np.round(np.nanpercentile(ser, 5), 2),
            'MIN': ser.min(),
            # 'RANGE': np.round(np.nanmax(ser) - np.nanmin(ser), 1),
            # 'IQR': np.round(np.nanpercentile(ser, 75) - np.nanpercentile(ser, 25), 1),
            # 'STD': np.round(np.nanstd(ser), 1),
            # 'VAR': np.round(np.nanvar(ser)),
            # ' ': ' ',
            # 'KURT.': np.round(ser.kurtosis(), 3),
            # 'SKEW': np.round(ser.skew(), 3),
            # 'SUM': np.round(np.nansum(ser), 3)
        })
    else:
        None
    
    return data

# data = np.random.randn(1000)  # Replace this with your dataset
# #compute(data,data).get('col')
# compute(data,data).get('data')
# compute(data,data).get('variable_type')
