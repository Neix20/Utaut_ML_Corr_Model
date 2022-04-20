import pandas as pd

def convert_nominal(arr, term_arr):
    tmp_dict = {val:ind for (ind, val) in enumerate(term_arr)}
    return arr.map(lambda x : tmp_dict[x])

def df_unique(arr):
    tmp_keys = []
    for val in arr:
        if type(val) != str:
            continue
        tmp_list = val.split(";")
        for val2 in tmp_list:
            if val2 not in tmp_keys:
                tmp_keys.append(val2)
                
    arr_list = []
    for val in arr:
        if type(val) != str:
            continue
        tmp_dict = {i:"No" for i in tmp_keys}
        tmp_list = val.split(";")
        for val2 in tmp_list:
            tmp_dict[val2] = "Yes"
        arr_list.append(tmp_dict)
    
    return pd.DataFrame(arr_list)