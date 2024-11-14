"""IMPORTS"""
import os
import sys
import importlib
import datetime as dt
import time
from pathlib import Path
from contextlib import redirect_stdout
import calendar

#import csPlots.py as csPlots

# Import data manipulation libraries
import numpy as np
import pandas as pd

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


"""READ FUNCTIONS"""

def read_in_df(df_path, names_path):
    """returns a df, dss_names in correct format
        df_path = path to extracted and converted data csv (ex. '../output/convert/convert_EDA_data_10_01_24.csv')
        names_path = path to extracted dss names csv (ex. '../data/metrics/EDA_10_01_24_dss_names.csv') 
    """
    df = pd.read_csv(df_path, header=[0, 1, 2, 3, 4, 5, 6], index_col=0, parse_dates=True)
    dss_names = pd.read_csv(names_path)["0"].tolist()
    return df, dss_names


"""SUBSET AND TRANSFORMATION FUNCTIONS"""

def add_water_year_column(df):
    df_copy = df.copy().sort_index()
    df_copy['Date'] = pd.to_datetime(df_copy.index)
    df_copy.loc[:, 'Year'] = df_copy['Date'].dt.year
    df_copy.loc[:, 'Month'] = df_copy['Date'].dt.month
    df_copy.loc[:, 'WaterYear'] = np.where(df_copy['Month'] >= 10, df_copy['Year'] + 1, df_copy['Year'])
    return df_copy.drop(["Date", "Year", "Month"], axis=1)

def create_subset_var(df, var):
    # should only use this function when units aren't considered
    """ 
    Filters df to return columns that contain the string varname
    :param df: Dataframe to filter
    :param varname: variable of interest, e.g. S_SHSTA
    """
    filtered_columns = df.columns.get_level_values(1).str.contains(var)
    return df.loc[:, filtered_columns]

def create_subset_unit(df, var, units):
    """ 
    Filters df to return columns that contain the string varname and units
    :param df: Dataframe to filter
    :param varname: variable of interest, e.g. S_SHSTA
    :param units: units of interest
    """
    var_filter = df.columns.get_level_values(1).str.contains(var)
    unit_filter = df.columns.get_level_values(6).str.contains(units)
    filtered_columns = var_filter & unit_filter
    return df.loc[:, filtered_columns]


"""MEAN, SD, IQR, SUM FUNCTIONS"""

def compute_mean(df, var, study_lst = None, units = "TAF", months = None):
    subset_df = create_subset_unit(df, var, units)
    if study_lst is not None:
        subset_df = subset_df.iloc[:, study_lst]
    
    subset_df = add_water_year_column(subset_df)
    
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]
        
    annual_mean = subset_df.groupby('WaterYear').mean()
    num_nonnull_yrs = annual_mean.dropna().shape[0]
    return (annual_mean.sum() / num_nonnull_yrs).iloc[-1]

def compute_sd(df, var, var_title, units = "TAF", months = None):
    subset_df = create_subset_unit(df, var, units)
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]

    standard_deviation = subset_df.std().to_frame(name=var_title).reset_index(drop=True)
    return standard_deviation

def compute_annual_sums(df, var, study_lst, units = "TAF", months = None):
    subset_df = create_subset_unit(df, var, units).iloc[:, study_lst]
    subset_df = add_water_year_column(subset_df)
    
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]
        
    annual_sum = subset_df.groupby('WaterYear').sum()

    return annual_sum

def compute_sum(df, var, study_lst, units = "TAF", months = None):
    df = compute_annual_sums(df, var, study_lst, units, months)
    return (df.sum()).iloc[-1]

def compute_iqr_value(df, iqr_value, var, var_title, study_list, units = "TAF", months=None, annual=True):
    if annual:
        subset_df = compute_annual_sums(create_subset_unit(df, var, units), var, study_list, units, months)
    else:
        subset_df = create_subset_unit(df, var, units)
        if months is not None:
            subset_df = subset_df[subset_df.index.month.isin(months)]
    iqr_values = subset_df.apply(lambda x: x.quantile(iqr_value), axis=0)
    iqr_df = pd.DataFrame(iqr_values, columns=['IQR']).reset_index()[["IQR"]].rename(columns = {"IQR": var_title})
    return iqr_df


"""EXCEEDANCE FUNCTIONS"""

def count_exceedance_days(data, threshold):
    """
    Count the number of days in the data that exceed a given threshold
    """
    exceedance_counts = pd.DataFrame(np.nan, index=[0], columns=data.columns)

    for col in data.columns:
        exceedance_counts.loc[0, col] = (data[col] > threshold).sum()
    return exceedance_counts

def calculate_flow_sum_per_year(flow_data):
    """
    Calculate the annual total of the given data per year
    :NOTE: This was translated from Abhinav's code and is only used in the exceedance_metric function
    """
    flow_data = add_water_year_column(flow_data)
    flow_sum_per_year = flow_data.groupby('WaterYear').sum(numeric_only=True).reset_index()

    return flow_sum_per_year

def calculate_exceedance_probabilities(df):
    exceedance_df = pd.DataFrame(index=df.index)
    for column in df.columns:
        sorted_values = df[column].dropna().sort_values(ascending=False)
        exceedance_probs = sorted_values.rank(method='first', ascending=False) / (1 + len(sorted_values))
        exceedance_df[column] = exceedance_probs.reindex(df.index)
    return exceedance_df

def exceedance_probability(df, var, threshold, month, var_title, units = "TAF"):
    # Subset data for the specific variable
    var_df = create_subset_unit(df, var, units)
    # Filter by the specified month and drop NaNs --> only valid values are used in calculating the exceedance probability
    var_month_df = var_df[var_df.index.month.isin([month])].dropna()
    # Count how often the values exceed the threshold and calculate the percentage
    result_df = count_exceedance_days(var_month_df, threshold) / len(var_month_df) * 100
    # Reshape the result to match the expected output format
    reshaped_df = result_df.melt(value_name=var_title).reset_index(drop=True)[[var_title]]
    return reshaped_df

def exceedance_metric(df, var, exceedance_percent, var_title, units = "TAF"):
    # Extract data for a specific variable in the desired units
    var_df = create_subset_unit(df, var, units)
    # Drop NaNs and calculate annual flow sums
    annual_flows = calculate_flow_sum_per_year(var_df).iloc[:, 1:].dropna()
    # Calculate exceedance probabilities for valid data only
    exceedance_probs = calculate_exceedance_probabilities(annual_flows)
    # Sort annual flows and exceedance probabilities for thresholding
    annual_flows_sorted = annual_flows.apply(np.sort, axis=0)[::-1]
    exceedance_prob_baseline = exceedance_probs.apply(np.sort, axis=0)
    if not exceedance_prob_baseline.empty:
        exceedance_prob_baseline = exceedance_prob_baseline.iloc[:, 0].to_frame()
        exceedance_prob_baseline.columns = ["Exceedance Sorted"]
    else:
        raise ValueError("No data available for exceedance probability calculation")
    # Find the index where exceedance probability meets or exceeds the given percentage
    #exceeding_index = exceedance_prob_baseline[exceedance_prob_baseline[‘Exceedance Sorted’] >= exceedance_percent].index[0]
    if 'Exceedance Sorted' not in exceedance_prob_baseline.columns:
        raise KeyError("Column 'Exceedance Sorted' not found in DataFrame")
    filtered_indices = exceedance_prob_baseline.loc[exceedance_prob_baseline['Exceedance Sorted'] >= exceedance_percent].index
    if len(filtered_indices) == 0:
        raise ValueError("No values found meeting the exceedance criteria")
    exceeding_index = filtered_indices[0]
    baseline_threshold = annual_flows_sorted.iloc[len(annual_flows_sorted) - exceeding_index - 1, 0]
    # Count exceedance days, ignoring NaNs
    result_df = count_exceedance_days(annual_flows, baseline_threshold).dropna() / len(annual_flows) * 100
    # Reshape the result for output format
    reshaped_df = result_df.melt(value_name=var_title).reset_index(drop=True)[[var_title]]
    return reshaped_df


"""CSUTOM METRIC FUNCTIONS"""
def calculate_monthly_average(flow_data):
    flow_data = flow_data.reset_index()
    flow_data['Date'] = pd.to_datetime(flow_data.iloc[:, 0])

    flow_data.loc[:, 'Month'] = flow_data['Date'].dt.strftime('%m')
    flow_data.loc[:, 'Year'] = flow_data['Date'].dt.strftime('%Y')

    flow_values = flow_data.iloc[:, 1:]
    monthly_avg = flow_values.groupby(flow_data['Month']).mean().reset_index()

    monthly_avg.rename(columns={'Month': 'Month'}, inplace=True)
    return monthly_avg

# Annual Avg (using dss_names)
def ann_avg(df, dss_names, var_name, units = "TAF"):
    metrics = []
    for study_index in np.arange(0, len(dss_names)):
        metric_value = compute_mean(df, var_name, [study_index], units, months=None)
        metrics.append(metric_value)

    ann_avg_delta_df = pd.DataFrame(metrics, columns=['Ann_Avg_' + var_name])
    return ann_avg_delta_df

# Annual X Percentile outflow of a Delta or X Percentile Resevoir Storage
def ann_pct(df, dss_names, pct, var_name, df_title, units = "TAF"):
    study_list = np.arange(0, len(dss_names))
    return compute_iqr_value(df, pct, var_name, df_title, study_list, units, months=None, annual=True)

# 1 Month Avg using dss_names
def mnth_avg(df, dss_names, var_name, mnth_num, units = "TAF"):
    metrics = []
    for study_index in np.arange(0, len(dss_names)):
        metric_value = compute_mean(df, var_name, [study_index], units, months=[mnth_num])
        metrics.append(metric_value)

    mnth_str = calendar.month_abbr[mnth_num]
    mnth_avg_df = pd.DataFrame(metrics, columns=[mnth_str + '_Avg_' + var_name])
    return mnth_avg_df

# All Months Avg Resevoir Storage or Avg Delta Outflow (based off plot_moy_averages)
def moy_avgs(df, dss_names, var_name, units = "TAF"):
    """
    The function assumes the DataFrame columns follow a specific naming
    convention where the last part of the name indicates the study. 
    """
    var_df = create_subset_unit(df, var_name, units)
    
    all_months_avg = {}
    for mnth_num in range(1, 13):
        metrics = []

        for study_index in np.arange(0, len(dss_names)):
            metric_val = compute_mean(var_df, var_name, [study_index], units, months=[mnth_num])
            metrics.append(metric_val)

        mnth_str = calendar.month_abbr[mnth_num]
        all_months_avg[mnth_str] = np.mean(metrics)
    
    moy_df = pd.DataFrame(list(all_months_avg.items()), columns=['Month', f'Avg_{var_name}'])
    return moy_df

# Monthly X Percentile Resevoir Storage or X Percentile Delta Outflow
def mnth_pct(df, dss_names, var_name, pct, df_title, mnth_num, units = "TAF"):
    study_list = np.arange(0, len(dss_names))
    return compute_iqr_value(df, pct, var_name, df_title, study_list, units, months = [mnth_num], annual = True)

def annual_totals(df, var_name, units = "TAF"):
    """
    Plots a time-series graph of annual totals for a given MultiIndex Dataframe that 
    follows calsim conventions
    
    The function assumes the DataFrame columns follow a specific naming
    convention where the last part of the name indicates the study. 
    """
    df = create_subset_unit(df, var_name, units)
    
    annualized_df = pd.DataFrame()
    var = '_'.join(df.columns[0][1].split('_')[:-1])
    studies = [col[1].split('_')[-1] for col in df.columns]
        
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in range(df.shape[1])]
    colors[-1] = [0,0,0,1]
        
    i=0
    
    for study in studies:
        study_cols = [col for col in df.columns if col[1].endswith(study)]
        for col in study_cols:
            with redirect_stdout(open(os.devnull, 'w')):
                temp_df = df.loc[:,[df.columns[i]]]
                temp_df["Year"] = df.index.year
                df_ann = temp_df.groupby("Year").sum()
                annualized_df = pd.concat([annualized_df, df_ann], axis=1)
                i+=1
                    
    return annualized_df

"""PAST VERSIONS OF FUNCTIONS (INCLUDE NAN VALUES)"""

"""def compute_mean(df, variable_list, study_lst, units, months = None):
    df = compute_annual_sums(df, variable_list, study_lst, units, months)
    num_years = len(df)
    return (df.sum() / num_years).iloc[-1]"""

"""def calculate_exceedance_probabilities(df):
    exceedance_df = pd.DataFrame(index=df.index)

    for column in df.columns:
        sorted_values = df[column].sort_values(ascending=False)
        exceedance_probs = (sorted_values.rank(method='first', ascending=False)) / (1 + len(sorted_values))
        exceedance_df[column] = exceedance_probs.sort_index()

    return exceedance_df"""

"""def exceedance_probability(df, var, threshold, month, vartitle):
    var_df = create_subset_var(df, var)
    var_month_df = var_df[var_df.index.month.isin([month])]
    result_df = count_exceedance_days(var_month_df, threshold) / len(var_month_df) * 100
    reshaped_df = result_df.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]
    return reshaped_df"""

"""
def exceedance_metric(df, var, exceedance_percent, vartitle, unit):
    var_df = create_subset_unit(df, var, unit)
    annual_flows = calculate_flow_sum_per_year(var_df).iloc[:, 1:]
    exceedance_probs = calculate_exceedance_probabilities(annual_flows)

    annual_flows_sorted = annual_flows.apply(np.sort, axis=0)[::-1]
    exceedance_prob_baseline = exceedance_probs.apply(np.sort, axis=0).iloc[:, 0].to_frame()
    exceedance_prob_baseline.columns = ["Exceedance Sorted"]

    exceeding_index = exceedance_prob_baseline[exceedance_prob_baseline['Exceedance Sorted'] >= exceedance_percent].index[0]
    baseline_threshold = annual_flows_sorted.iloc[len(annual_flows_sorted) - exceeding_index - 1, 0]

    result_df = count_exceedance_days(annual_flows, baseline_threshold) / len(annual_flows) * 100
    reshaped_df = result_df.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]

    return reshaped_df"""

"""def compute_mean(df, var, study_lst = None, units = "TAF", months = None):
    df = compute_annual_means(df, var, study_lst, units, months)
    num_nonnull_yrs = df.dropna().shape[0]
    return (df.sum() / num_nonnull_yrs).iloc[-1]"""

"""def compute_annual_means(df, var, study_lst = None, units = "TAF", months = None):
    subset_df = create_subset_unit(df, var, units)
    if study_lst is not None:
        subset_df = subset_df.iloc[:, study_lst]
    
    subset_df = add_water_year_column(subset_df)
    
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]
        
    annual_mean = subset_df.groupby('WaterYear').mean()
    return annual_mean"""
    