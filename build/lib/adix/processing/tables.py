import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from jinja2 import Template
from IPython.display import HTML
import base64
from io import BytesIO
import pandas as pd


from ..datahub import DataHub
from ..dtype import *


style = """
.my-custom-styles body {
    font-family: Arial, sans-serif;
}

.container {
    padding: 10px;
    margin: 1px;
    box-sizing: border-box;
}

.flex-container {
    display: flex;
    flex-wrap: wrap;  /* Add flex-wrap for responsive layout */
}

.flexing {
    flex-grow: 1;
}

.con-100 {
    width: 10px;
}

.container-200 {
    width: 200px;  /* Set the desired width directly */
}

.container-300 {
    width: 350px;  /* Set the desired width directly */
}

.container-400 {
    width: 450px;  /* Set the desired width directly */
}

.container-500 {
    width: 488px;  /* Set the desired width directly */
}

.container-800 {
    width: 800px;  /* Set the desired width directly */
}

.container-1000 {
    width: 1000px;  /* Set the desired width directly */
}

.img-container {
    width: 100%;
    text-align: center;
}

.img-thumb {
    overflow: hidden;
    height: auto;
    width: 100%;
}

.mytable {
    /* Consider removing float: left; and width: 100%; if not needed */
    font-family: Tahoma, Algerian, Geneva, sans-serif;
    width: 100%;
    border-collapse: collapse;
}

.mytable td {
    padding: 8px;  /* Adjusted padding for better spacing */
    border: 1px solid #dddfe1;
    border-left: none;
    border-right: none;
    height: 34px;  /* Set your desired height for the table cells */
    white-space: nowrap !important;  /* Prevent text from wrapping */
}

.mytable thead td {
    text-align: left;
    background-color: #386087;
    color: #ffffff;
    font-weight: bold;
    font-size: 13px;
    border: 1px solid #54585d;
}

.mytable tbody td {
    text-align: left !important;
    color: #636363;
}

.mytable tbody tr:nth-child(even) {
    background-color: white;
}

.mytable tbody tr:nth-child(even):hover {
    background-color: #e3f2fd;
}

.mytable tbody tr td:first-child {
    font-weight: bold;
}

"""
def create_container_html(container_name, container_keys, data, cfg):
    hover_color = cfg.get('hover_color', '#e3f2fd')  # Default hover color
    
    # Set the widths based on the container_name
    con_size = '200'
    if container_name == 'Stats':
        con_size = '300'
        stat_width = '220px'  # 1/4 width
        value_width = '80px'  # 3/4 width
        
    elif container_name == 'Alerts':
        con_size = '400'
        stat_width = '370px'  # 1/4 width
        value_width = '80px'  # 3/4 width
    
    elif container_name == 'container3a':
        stat_width = '25%'  # 1/4 width
        value_width = '75%'  # 3/4 width
    else:
        stat_width = '50%'  # Default width for other containers
        value_width = '50%'


    # Check if container_name is in the specified list
    if container_name in ['Alerts', 'Stats', 'Columns']:
        # Add table headers based on the condition
         table_headers = f"<th style='width: {stat_width}; text-align: center !important; font-size: larger; color: #333;'>{container_name}</th>"
    else:
        # If container_name is not in the list, initialize an empty string for table_headers
        table_headers = ""

    table_rows = []
    # Iterate through the data items
    for stat, value in data.items():
        if stat in container_keys:
            if container_name == 'Alerts':
                # If the value is a dictionary, iterate through its items and create additional table rows
                table_rows.extend(f"<tr><td style='width: {stat_width};'>"
                                  f"<span style='color: #FF6961;'>{inner_stat}</span> has {inner_value[0]} {stat} {inner_value[1]}."
                                  #f"</td><td style='width: {value_width}; text-align: center !important;'>{inner_value[2]} </td></tr>" # Alert
                                  for inner_stat, inner_value in value.items())
            else:
                # If the value is not a dictionary, create a regular table row
                table_rows.append(f"<tr><td style='width: {stat_width};'>{stat}</td><td style='width: {value_width}; text-align: center !important;'>{value}</td></tr>")

    return f"""
        <div class="container container-{con_size}">
            <table class="mytable">
                {table_headers}
                {"".join(table_rows)}
            </table>
        </div>
    """


def create_main_html(template_title, style, containers_html, pics, cfg):
    hover_color = cfg.get('hover_color', '#e3f2fd') #default
    img_width = "600px" if template_title == "val_tableC" else "500px"
    img_pad = "0px" if template_title == "val_tableC" else "0px"
    image_section = f"""
        <div class="container-500">
            <div class="img-container-cat" style="width:{img_width}; padding-left: {img_pad};">
                <img class="img-thumb-cat" style="" src="data:image/png;base64, {pics[0]}" alt="Histogram">
            </div>
        </div>
    """ if template_title != "val_tableW" else ""

    return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{template_title}</title>
            <style>{style}</style>
            <style>
                .mytable tbody tr:nth-child(even):hover {{
                    background-color: {hover_color} !important;
                }}
                .mytable tbody tr:nth-child(odd):hover {{
                    background-color: {hover_color} !important;
                }}
            </style>
        </head>
        <body class="my-custom-styles">
            <div class="flex-container" style="flex-wrap: wrap;">
                {containers_html}

                {image_section}
            </div>

            <div class="flex-container">

            </div>
            <!-- New container with bottom border -->
            <div class="container-gen" style="width: 100%; border-bottom: 1px solid #DDDFE1;"></div>
        </body>
        </html>
    """


def generate_table(data,container_keys,table_name, style=None, *pics, cfg=None):
    cfg = cfg or {}
    containers_html = "".join(create_container_html(container_name, container_keys, data, cfg) for container_name, container_keys in container_keys.items())
    return create_main_html(table_name, style, containers_html, pics, cfg)


##############################################################################################################################


def value_tableX(col,cfg,tokens=None, df=None):

    if tokens is not None:
        #dat = tokens
        dat = col.value_counts()
    else:
        dat = col.value_counts()


    # Separate the data into the first 20 rows and the rest
    top_20 = dat.head(10)
    other_values = dat.iloc[10:]

    # Create rows for the first 20 values
    rows_top_20 = [f"<tr><td>{idx[:20] if isinstance(idx, str) else idx}</td><td style='text-align: left !important;'>{val}</td><td>{valp}</td></tr>" for idx, val, valp in zip(top_20.index, top_20.values, np.round((top_20.values / col.size) * 100, 2))]

    # Create a row for the other values
    row_other_values = f"<tr><td>Other Values</td><td style='text-align: left !important;'>{other_values.sum()}</td><td>{np.round((other_values.sum() / col.size) * 100, 2)}</td></tr>"

    # Concatenate all rows
    all_rows = "".join(rows_top_20) + row_other_values

    top3 = col.head(3)
    low3 = col.tail(3)

    hover_color = cfg.get('hover_color','#e3f2fd')

    
    # Create a value table with f-strings (modify this as needed)
    value_table = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>val_tableC</title>
            <style>
                .mycs body {{
                    font-family: Arial, sans-serif;
                }}

                .containerX {{
                    padding: 10px;
                    margin: 1px;
                    box-sizing: border-box;
                }}

                .flex-containerX {{
                    display: flex;
                    flex-wrap: wrap;
                }}

                .container-200X {{
                    flex: 2;
                }}

                .mytableX {{
                    float: left;
                    width: 100%;
                    font-family: Tahoma, Algerian, Geneva, sans-serif;
                    border-collapse: collapse;
                }}

                .mytableX td {{
                    padding: 8px;
                    border: 1px solid #dddfe1;
                    border-left: none;
                    border-right: none;
                    height: 34px;
                    white-space: nowrap !important;
                }}

                .mytableX thead td {{
                    text-align: left !important;
                    background-color: #ffffff; /* White background color */
                    color: #386087; /* Your desired text color */
                    font-weight: bold;
                    font-size: 13px;
                    padding: 10px; /* Adjusted padding for better spacing */
                }}

                .mytableX tbody td {{
                    text-align: left !important;
                    color: #636363;
                }}

                .mytableX tbody tr:nth-child(even) {{
                    background-color: white;
                }}

                .mytableX tbody tr:nth-child(even):hover {{
                    background-color: {hover_color} !important;
                }}

                .mytableX tbody tr:nth-child(odd):hover {{
                    background-color: {hover_color} !important;
                }}

                .mytableX tbody tr td:first-child {{
                    font-weight: bold;
                }}
            </style>
        </head>
         <body class="mycs">
            <div class="flex-containerX">
                <div class="containerX container-200X">
                   <table class="mytableX">
                        <thead>
                            <tr>
                                <td>Values</td>
                                <td>Count</td>
                                <td>Frequency (%)</td>
                            </tr>
                        </thead>
                        {all_rows}
                    </table>
                </div>
            <div class="flex-containerX">
                <div class="containerX container-200X" style="width: 600px;">
                <div>
                <table class="mytableX" style="margin-bottom: 34px !important;">
                        <thead>
                            <tr>
                                <td style="width: 150px;">Head</td>
                                <td>Values</td>
                               
                            </tr>
                        </thead>
                        {''.join(f'<tr><td style="width: 150px;">{stat}</td><td style="text-align: left;">{value}</td></tr>' for stat, value in top3.items())}
                    </table>
                     </div>
                     <table class="mytableX">
                        <thead>
                            <tr>
                                <td style="width: 150px;">Tail</td>
                                <td>Values</td>
                                
                            </tr>
                        </thead>
                        {''.join(f'<tr><td style="width: 150px;">{stat}</td><td style="text-align: left;">{value}</td></tr>' for stat, value in low3.items())}
                    </table>
                     </div>
                    </div>
                     
            </div>
            <div class="container-x" style="width: 100%; border-bottom: 1px solid #DDDFE1;"></div>
            </div>
        </body>
        </html>
    """
    return value_table

    
def value_tableW(df,cfg):
    # Convert the first and last 10 rows of the DataFrame to HTML
    df_html_head = df.head(10).to_html(index=False, justify='left', escape=False)
    df_html_tail = df.tail(10).to_html(index=False, justify='left', escape=False)

    hover_color = cfg.get('hover_color','#e3f2fd')
    num = cfg.get('w_get_rows',5)

    # Create a value table with basic styling
    value_table = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DataFrame Value Table</title>
            <style>
                .dfw tbody tr:nth-child(even):hover {{
                    background-color: {hover_color} !important;
                }}
                .dfw tbody tr:nth-child(odd):hover {{
                    background-color: {hover_color} !important;
                }}
                </style>
           
        </head>
        <body>
            <div class="dfw">
                <h2>Top 10 Rows</h2>
                {df_html_head}
                <h2>Bottom 10 Rows</h2>
                {df_html_tail}
            </div>
        </body>
        </html>
    """
    return value_table


