from pathlib import Path
import json
import os
import numpy as np
import pandas as pd


from .dtype import *


class DataHub(dict):
    """This class contains DataHub results."""

    def __init__(self, *args, **kwargs):
        if 'variable_type' in kwargs:
            
            super().__init__(**kwargs)
            self.variable_type = kwargs["variable_type"]
        else:
            raise ValueError("Unsupported initialization, missing visula type")


       
