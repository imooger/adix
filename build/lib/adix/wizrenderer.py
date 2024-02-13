import random
import sys

from jinja2 import Environment, PackageLoader
from jinja2 import Template
from IPython.display import HTML

from .configs import *



class WizRenderer:
    def __init__(self, data_load, variable_type, cfg):
        #print(to_render)
        """
        {'type': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'], 
        'title': [None, None, None, None], 
        'image': [None, None, None, None], 
        'value_table': [None, None, None, None]}
        """
        self.cfg = cfg
        self.context = DataManager(**data_load)
        self.template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Matplotlib Plots</title>
    <style>
        /* Reset CSS -> Use a Reset CSS: to ensure consistent default styles across different browsers. */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        /* Shared styles for tabset */
        .tabset-{{ context.unique_id }} {
            max-width: 1118px;
            padding-top: 15px;
        }

        .tabset-{{ context.unique_id }} > input[type="radio"] {
            position: absolute;
            left: -200vw;
        }

        .tabset-{{ context.unique_id }} .tab-panel {
            display: none;
            padding-top: 10px;
            border-top: 1px solid #ccc;
            width: 1100px;
        }

        .tabset-{{ context.unique_id }} > input:checked ~ .tab-panels > .tab-panel {
            display: block;
        }

        .tabset-{{ context.unique_id }} > label {
            position: relative;
            display: inline-block;
            padding: 15px 15px 25px;
            border: 1px solid transparent;
            border-bottom: 0;
            cursor: pointer;
            font-weight: 600;
            height: 64px;
            color: #525252;
        }

        input:focus-visible + label {
            outline: 2px solid rgba(0, 102, 204, 1);
            border-radius: 3px;
        }

        .tabset-{{ context.unique_id }} > label:hover,
        .tabset-{{ context.unique_id }} > input:focus + label,
        .tabset-{{ context.unique_id }} > input:checked + label {
           color: {{ cfg.label_color }};
        }

        .tabset-{{ context.unique_id }} > label:hover::after,
        .tabset-{{ context.unique_id }} > input:focus + label::after,
        .tabset-{{ context.unique_id }} > input:checked + label::after {
            background: #06c;
        }

        .tabset-{{ context.unique_id }} > input:checked + label {
            border-color: #ccc;
            border-bottom: 1px solid #fff;
            margin-bottom: -1px;
            height: 65px;
        }

        /* Unique body styling */
        .body-{{ context.unique_id }} {
            font: 16px/1.5em "Overpass", "Open Sans", Helvetica, sans-serif;
            color: #333;
            font-weight: 300;
            padding: 30px;
        }

    </style>
</head>
<body class="body-{{ context.unique_id }}">
    <div class="tabset-{{ context.unique_id }}">
        {% for idx in range(context.type|length) %}
            {% if context.title[idx] %}
                <input type="radio" name="tabset-{{ context.unique_id }}" id="{{ context.type[idx] }}-{{ context.unique_id }}" aria-controls="{{ context.type[idx] }}-{{ context.unique_id }}" {% if loop.first %}checked{% endif %}>
                <label for="{{ context.type[idx] }}-{{ context.unique_id }}">{{ context.title[idx] }}</label>
            {% endif %}
        {% endfor %}

        <div class="tab-panels">
            {% for idx in range(context.type|length) %}
                <section id="{{ context.type[idx] }}-{{ context.unique_id }}" class="tab-panel" {% if not loop.first %}style="display: none;"{% endif %}>
                    {% if context.value_table[idx] is not none %}
                        {{ context.value_table[idx]|safe }}
                    {% elif context.image[idx] is not none %}
                        {% for image_data in context.image[idx] %}
                            {% set image_width = image_data[1] | default(500) %}
                            <img src="data:image/png;base64, {{ image_data[0] }}" alt="{{ context.title[idx] }}" style="width: {{ image_width }}px; margin-right: 0px;">
                        {% endfor %}
                    {% endif %}
                </section>
            {% endfor %}
        </div>

        <script>
            {% for idx in range(context.type|length) %}
                document.getElementById("{{ context.type[idx] }}-{{ context.unique_id }}").addEventListener("change", function() {
                    showPanel("{{ context.type[idx] }}-{{ context.unique_id }}");
                });
            {% endfor %}

            function showPanel(panelId) {
                const panel = document.getElementById(panelId);
                const panels = panel.parentElement.querySelectorAll('.tab-panel');
                panels.forEach(p => {
                    p.style.display = p.id === panelId ? "block" : "none";
                });
            }
        </script>
    </div>
</body>
</html>
        """)


    def _repr_html_(self):
        output_html = self.template.render(context=self.context, cfg=self.cfg)
        return output_html



    def show(self):
        """
        Render the report. This is useful when calling plot in a for loop.
        """
        display(HTML(self._repr_html_()))

    
class DataManager:
    def __init__(self, **data_load):
        for key, value in data_load.items():
            setattr(self, key, value)
        self.unique_id = random.randint(1, 10000)
