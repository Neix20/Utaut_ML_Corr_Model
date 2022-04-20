import numpy as np
import pandas as pd
import itertools as itr

from Utilities.CFS import *

def pairwise_correlation(df, func):
    columns = [*range(df.shape[1])]
    
    arr_list = []
    
    for col_a, col_b in itr.combinations(columns, 2):
        corr, p_val = func(df.iloc[:, col_a], df.iloc[:, col_b])
        arr_list.append([col_a, col_b, corr])
#         arr_list.append([col_b, col_a, corr])
        
    # Sort List by First Variable
    arr_list = sorted(arr_list, key = lambda x : x[0])
        
    result = pd.DataFrame(arr_list, columns = ["Var A", "Var B", "correlation"])
    
    # Replace Variable Names
    result["Var A"] = [df.columns[col_ind] for col_ind in result["Var A"]]
    result["Var B"] = [df.columns[col_ind] for col_ind in result["Var B"]]
    
    return result

def create_corr_matrix(df, func):
    columns = [*range(df.shape[1])]

    arr_list = []
    
    for col_a in columns:
        corr, p_val = func(df.iloc[:, col_a], df.iloc[:, col_a])
        arr_list.append([col_a, col_a, corr])

    for col_a, col_b in itr.combinations(columns, 2):
        corr, p_val = func(df.iloc[:, col_a], df.iloc[:, col_b])
        arr_list.append([col_a, col_b, corr])
        arr_list.append([col_b, col_a, corr])
        
    # Sort List by First Variable
    arr_list = sorted(arr_list, key = lambda x : x[0])
        
    result = pd.DataFrame(arr_list, columns = ["Var A", "Var B", "correlation"])
    result = result.pivot(index = "Var A", columns = "Var B", values = "correlation")
    
    result.index = df.columns
    result.columns = df.columns
    
    return result

def create_corr_metric_matrix(df, func):
    columns = [*range(df.shape[1])]

    arr_list = []

    for col_a in columns:
        corr = merit_calculation(df.iloc[:, [col_a]], df.iloc[:, col_a], func)
        arr_list.append([col_a, col_a, corr])

    for col_a, col_b in itr.combinations(columns, 2):
        corr = merit_calculation(df.iloc[:, [col_a]], df.iloc[:, col_b], func)
        arr_list.append([col_a, col_b, corr])
        arr_list.append([col_b, col_a, corr])

    # Sort List by First Variable
    arr_list = sorted(arr_list, key = lambda x : x[0])

    result = pd.DataFrame(arr_list, columns = ["Var A", "Var B", "correlation"])
    result = result.pivot(index = "Var A", columns = "Var B", values = "correlation")

    result.index = df.columns
    result.columns = df.columns

    return result