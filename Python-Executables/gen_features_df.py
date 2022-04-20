import sys
import json
import pandas as pd

filename = sys.argv[1]
filter_feature = sys.argv[2]
targeted_feature = sys.argv[3]
utaut_feature = sys.argv[4]

filter_feature = filter_feature.split("->")

if filter_feature == ["None"]:
    filter_feature = []

targeted_feature = targeted_feature.split("->")
utaut_feature = utaut_feature.split("->")

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


# Read Dataset
df = pd.read_csv(f"Dataset\\{filename}")

# Generate Targeted Feature DataFrame
targeted_feature_df = df.loc[:, targeted_feature]

# Generate UTAUT Factor Dataframe
utaut_feature_df = df.loc[:, utaut_feature]

# Generate Filter Feature Dataframe
filter_feature_df = df.loc[:, filter_feature]

# Initialize Variable
filter_feature_res_df = pd.DataFrame()
arr_df = []
feature_dict = {}

# Make new Columns of Yes-No for Every unique Values
for feat in filter_feature_df.columns:
    tmp_df = df_unique(filter_feature_df.loc[:, feat])
    if len(tmp_df.columns.tolist()) == 2:
        filter_feature_res_df[feat] = filter_feature_df.loc[:, feat]
        feature_dict[feat] = ["None"]
    else:
        feature_dict[feat] = tmp_df.columns.tolist()
        tmp_df.columns = [f"{feat}->{val}" for val in tmp_df.columns]
        arr_df.append(tmp_df)
        
feature_dict["None"] = ["None"]
        
filter_feature_res_df.columns = [f"{feat}->None" for feat in filter_feature_res_df.columns]
        
arr_df.append(filter_feature_res_df)

# Generate Filter Feature Res DF
filter_feature_res_df = pd.concat(arr_df, axis = 1)

# Combine Targeted Feature, Filter Feature and UTAUT Feature
arr_df = [targeted_feature_df, filter_feature_res_df, utaut_feature_df]
df = pd.concat(arr_df, axis = 1)

# Output Dataframe
df.to_csv("Dataset\\res_processed.csv", index = False)

print("Sucessfully output res_processed.csv")

# Output JSON
with open("Dataset\\feat_dict.json", 'w') as fout:
    json_dumps_str = json.dumps(feature_dict, indent=4)
    print(json_dumps_str, file = fout)
    
print("Successfully output feat_dict.json")

sys.stdout.flush()