from coeqwalpythonlib import myfunctions

import os
import sys
import importlib
import datetime as dt
import time
from pathlib import Path
from contextlib import redirect_stdout
import calendar

# Import data manipulation libraries
import numpy as np
import pandas as pd

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

"""def test_read_in_df():
    #df = pd.read_csv('/Users/clairemarie/Desktop/CoEqwal/Correct Folder Nesting/output/convert/convert_EDA_data_10_01_24.csv', header=[0, 1, 2, 3, 4, 5, 6], index_col=0, parse_dates=True)
    #dss_names = pd.read_csv('/Users/clairemarie/Desktop/CoEqwal/Correct Folder Nesting/data/metrics/EDA_10_01_24_dss_names.csv')["0"].tolist()

    df = myfunctions.read_in_df('/Users/clairemarie/Desktop/CoEqwal/Correct Folder Nesting/output/convert/convert_EDA_data_10_01_24.csv', '/Users/clairemarie/Desktop/CoEqwal/Correct Folder Nesting/data/metrics/EDA_10_01_24_dss_names.csv')
    df = pd.DataFrame({1: [10], 2: [20]})
    exactly_equal = pd.DataFrame({1: [10], 2: [20]})
    assert  df.equals(exactly_equal)"""

df,dss_names = myfunctions.read_in_df('/Users/clairemarie/Desktop/CoEqwal/Correct Folder Nesting/output/convert/convert_EDA_data_10_01_24.csv', '/Users/clairemarie/Desktop/CoEqwal/Correct Folder Nesting/data/metrics/EDA_10_01_24_dss_names.csv')

def test_ann_avg():
    output_df = myfunctions.compute_mean(df, )