import numpy as np
import pandas as pd

from Utilities.accuracy import *

def cmp_n_feature_df(X, Y, model, func, s_num = 5, e_num = 16):
    arr_list = []
    
    for num in range(s_num, e_num):
        feature_set = func(X, Y, model, num)
    
        acc_score = get_acc_score_kcv(X, Y, model)
    
        arr_list.append((num, feature_set, round(acc_score, 2)))
        
    return pd.DataFrame(arr_list, columns = ["Number of Features", "Feature Set", "Accuracy Score"])

def feature_prob_df(feature_set, ori_arr):
    tmp_dict = {col:0 for col in ori_arr}
    
    for feature_arr in feature_set:
        for feature in feature_arr:
            tmp_dict[feature] += 1
            
    tmp_df = pd.DataFrame(pd.Series(tmp_dict), columns = ["Feature"])
    tmp_df = tmp_df.sort_values(by = ["Feature"], ascending = False)
            
    return tmp_df