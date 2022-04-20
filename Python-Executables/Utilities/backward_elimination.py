import numpy as np
import pandas as pd

from Utilities.accuracy import *

def backward_selection(X, Y, clf, n_selected_features = 10):
    # Number of Features
    n_features = X.shape[1]
    
    # selected feature set, initialized to be empty
    count, feature_set = n_features, [*range(n_features)]

    while count > n_selected_features:
        
        f_loop = [i for i in range(n_features) if i in feature_set]
        
        # 1. Get All Accuracy Score for All Subset in Feature Set
        acc_arr = []
        
        for i in f_loop:
            feature_set.remove(i)
            acc_arr.append(get_acc_score_kcv(X.iloc[:, feature_set], Y, clf))
            feature_set.append(i)
            
        # 2. Convert Python List to Series
        acc_arr = pd.Series(acc_arr, index = f_loop)
        
        # 3. Get The Largest idx
        idx = acc_arr.idxmax()
        
        # 4. delete the feature which results in the largest accuracy
        feature_set.remove(idx)
        
        count -= 1
        
    # Sort Feature Set
    feature_set = sorted(feature_set)
    
    # Get Feature from X.Columns
    feature_set = list(map(lambda x: X.columns[x], feature_set))
        
    return feature_set

def backward_selection_p_val(X, Y, func, n_selected_features = 10):
    # Number of Features
    n_features = X.shape[1]
    
    # selected feature set, initialized to be empty
    count, feature_set = n_features, [*range(n_features)]
    
    while count > n_selected_features:
        
        f_loop = [i for i in range(n_features) if i in feature_set]
    
        # 1. Get (Coeff, P_val) for Each Columns in F_Loop
        p_arr = [func(X.iloc[:, i], Y) for i in f_loop]
    
        # 2. Filter and Get only P_Val
        p_arr = map(lambda x: x[1], p_arr)
    
        # 3. Convert p_arr into Series
        p_arr = pd.Series(p_arr, index = f_loop)
    
        if max(p_arr) > 0.05:
            feature_set.remove(p_arr.idxmax())
            count -= 1
        else:
            break
    
    # Sort Feature Set
    feature_set = sorted(feature_set)
    
    # Get Feature from X.Columns
    feature_set = list(map(lambda x: X.columns[x], feature_set))
        
    return feature_set