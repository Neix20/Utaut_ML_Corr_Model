import sys
import time
import json
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr, chi2, chi2_contingency
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef

from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

from dtreeviz.trees import *

from Utilities.CFS import *
from Utilities.accuracy import *
from Utilities.corr_matrix import *
from Utilities.utils import *

from Visualization.model_graph import *
from Visualization.network_graph import *

filter_feature = sys.argv[1]
target_feature = sys.argv[2]
utaut_feature = sys.argv[3]

# Split String into Array
utaut_feature = utaut_feature.split("->")

if filter_feature != "None->None":
    feature_arr = [filter_feature, target_feature] + utaut_feature
else:
    feature_arr = [target_feature] + utaut_feature

print("Started Operation")

# Read Dataset
df = pd.read_csv("Dataset\\res_processed.csv")

# Create New Dataset From Selected Features
df = df.loc[:, feature_arr]

if filter_feature != "None->None":
    # Filter Values Based on "Yes" and "No"
    df = df[df[filter_feature] == "Yes"]

    # Drop Filter Feature
    df = df.drop([filter_feature], axis = 1)

# Output CSV
df.to_csv("Dataset\\train_test_df.csv", index=False)

print("Successfully output train_test_df.csv!")

# Split Dataset
df_X = df.iloc[:, 1:]
df_Y = df.iloc[:, 0]

# Convert Nominal
df_Y_unique_arr = df_Y.unique().tolist()
df_Y = convert_nominal(df_Y, df_Y_unique_arr)

# Convert df_Y into Int Datatype
df_Y = df_Y.astype(int)

# Remove Insignificant features using Chi-Square
arr_list = []

prob = 0.95

# Get List of P_Values
for col in df_X.columns:
    chi_df = pd.crosstab(df_X.loc[:, col], df_Y)
    stat, p, dof, expected = chi2_contingency(chi_df)
    
    critical = chi2.ppf(prob, dof)
    
    alpha = 1 - prob
        
    if abs(stat) >= critical and p <= alpha:
        arr_list.append((col, stat))
        
# Sort Variables Ascending by P_val
arr_list = sorted(arr_list, key = lambda x : x[1])

arr_df = pd.DataFrame(arr_list, columns = ["Variables", "Chi-Square Value"])

if arr_df.shape[0] > 4:
    df_X = df_X.loc[:, arr_df["Variables"]]
    df_Y = df_Y

print("Completed Removing Features using Chi-Square!")

# Select Intersected, Non-Intersected, Union Features using CFS (Self-Made Library)
func_arr = [pearsonr, spearmanr]

tmp_X = df_X
tmp_Y = df_Y

cfs_dict = {}

# Select Intersected Features
corr_dict = {}

name = "inter_feature_set"
corr_dict["feature_set"] = []
corr_dict["corr_dict"] = {}

inter_feature_set = CFS_Intersection(tmp_X, tmp_Y, func_arr)

if len(inter_feature_set) > 0:
    tmp_df = df_X.loc[:, inter_feature_set]

    tmp_df.columns = [name.split(":")[0] for name in tmp_df.columns]
    corr_df = create_corr_metric_matrix(tmp_df, pearsonr)

    corr_df = corr_df.where(np.triu(np.ones(corr_df.shape)).astype(np.bool))
    
    corr_df = corr_df.replace({np.nan: None})
    
    corr_dict["feature_set"] = list(inter_feature_set)
    corr_dict["corr_dict"] = corr_df.to_dict()
    
cfs_dict[name] = corr_dict

# Select Non-intersected Features
corr_dict = {}

name = "non_inter_feature_set"
corr_dict["feature_set"] = []
corr_dict["corr_dict"] = {}

non_inter_feature_set = CFS_Non_Intersection(tmp_X, tmp_Y, func_arr)

if len(non_inter_feature_set) > 0:
    tmp_df = df_X.loc[:, non_inter_feature_set]

    tmp_df.columns = [name.split(":")[0] for name in tmp_df.columns]
    corr_df = create_corr_metric_matrix(tmp_df, pearsonr)

    corr_df = corr_df.where(np.triu(np.ones(corr_df.shape)).astype(np.bool))
    
    corr_df = corr_df.replace({np.nan: None})
    
    corr_dict["feature_set"] = list(non_inter_feature_set)
    corr_dict["corr_dict"] = corr_df.to_dict()
    
cfs_dict[name] = corr_dict

# Select Union Features
name = "union_feature_set"

corr_dict = {}
corr_dict["feature_set"] = []
corr_dict["corr_dict"] = {}

union_feature_set = CFS_Union(tmp_X, tmp_Y, func_arr)

if len(union_feature_set) > 0:
    tmp_df = df_X.loc[:, union_feature_set]

    tmp_df.columns = [name.split(":")[0] for name in tmp_df.columns]
    corr_df = create_corr_metric_matrix(tmp_df, pearsonr)

    corr_df = corr_df.where(np.triu(np.ones(corr_df.shape)).astype(np.bool))
    
    corr_df = corr_df.replace({np.nan: None})
    
    corr_dict["feature_set"] = list(union_feature_set)
    corr_dict["corr_dict"] = corr_df.to_dict()

cfs_dict[name] = corr_dict
    
print("Completed Selecting Features using CFS!")
    
# Apply SMOTE imbalance to balance out imbalanced class
# Check if Targeted Variable have any imbalanced Class
tmp_Y = df_Y

imb_df = pd.DataFrame(index = df_Y_unique_arr)

imb_df["Count"] = tmp_Y.value_counts().sort_index().set_axis(df_Y_unique_arr)

imb_df["Count (%)"] = round(imb_df["Count"] / imb_df["Count"].sum() * 100.0, 2)

if min(imb_df["Count"]) > len(df_Y_unique_arr) * 2:
    oversample = SMOTE()
    df_X, df_Y = oversample.fit_resample(df_X, df_Y)
    
print("Completed SMOTE Imbalance!")

# Train Model Using Decision Tree Classifier
class ModelObj:
    def __init__(self, model, name, accuracy, clf_report, confusion_matrix, mcc, time_taken):
        self.model = model
        self.name = name
        self.accuracy = accuracy
        self.clf_report = clf_report
        self.confusion_matrix = confusion_matrix
        self.mcc = mcc
        self.time_taken = time_taken
        
def create_ModelObj(model, name, X, Y, class_arr):
    
    # Time Taken
    start = time.process_time()
    
    # Train Test Split5
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    
    # Train Model
    model.fit(X_train, y_train)
    
    # Get Y Predict
    y_pred = model.predict(X_test)
    
    # Accuracy
    acc_score = get_acc_score_kcv(X_train, y_train, model)

    # Classification Report
    tf_dict = { str(ind):val for ind, val in enumerate(class_arr)}
    clf_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict = True))
    clf_report.rename(tf_dict, axis = 1, inplace=True)
    clf_report = clf_report.T

    # Confusion Matrix
    tf_dict = { ind:val for ind, val in enumerate(class_arr)}
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    confusion_matrix_df.rename(tf_dict, axis = 0, inplace=True)
    confusion_matrix_df.rename(tf_dict, axis = 1, inplace=True)
    
    # Matthew Correlation Coefficient
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # your code here 
    time_taken = time.process_time() - start
    
    return ModelObj(model, name, acc_score, clf_report, confusion_matrix_df, mcc, time_taken)

# Initialize Model Dictionary
model_dict = {}

# Decision Tree (All Features)
tmp_X = df_X
tmp_Y = df_Y

model = DecisionTreeClassifier(max_depth = 3)
name = "Decision Tree"

model_dict[name] = create_ModelObj(model, name, tmp_X, tmp_Y, df_Y_unique_arr)

# Generate Decision Tree Visualization using DGraphViz
viz_col = [name.split(":")[0] for name in tmp_X.columns]

# Output Name into a new JSON File

viz = dtreeviz(
    model_dict[name].model, 
    tmp_X, 
    tmp_Y, 
    feature_names = viz_col, 
    class_names = df_Y_unique_arr,
    fancy = False)

viz.save("public\\img\\decision_tree_viz.svg")

# Decision Tree (Intersected Features)
if len(inter_feature_set) > 0:
    tmp_X = df_X.loc[:, inter_feature_set]
    tmp_Y = df_Y

    model = DecisionTreeClassifier()
    name = "Decision Tree (IFS)"

    model_dict[name] = create_ModelObj(model, name, tmp_X, tmp_Y, df_Y_unique_arr)

# Decision Tree (Non-Intersected Features)
if len(non_inter_feature_set) > 0:
    tmp_X = df_X.loc[:, non_inter_feature_set]
    tmp_Y = df_Y

    model = DecisionTreeClassifier()
    name = "Decision Tree (NIFS)"

    model_dict[name] = create_ModelObj(model, name, tmp_X, tmp_Y, df_Y_unique_arr)

# Decision Tree (Union Features)
if len(union_feature_set) > 0:
    tmp_X = df_X.loc[:, union_feature_set]
    tmp_Y = df_Y

    model = DecisionTreeClassifier()
    name = "Decision Tree (UFS)"

    model_dict[name] = create_ModelObj(model, name, tmp_X, tmp_Y, df_Y_unique_arr)

print("Completed Training Model!")

# Result Visualization

# Result DataFrame
m_arr = [(name, model_dict[name].clf_report, model_dict[name].accuracy, model_dict[name].mcc, model_dict[name].time_taken) for name in model_dict]
res_df = cmp_result_tbl(m_arr, "weighted avg")

# Precision Graph
clf_report_arr = [(name, model_dict[name].clf_report) for name in model_dict]
tmp_df = get_df_type(clf_report_arr, "Precision")
fig = pfr_graph(tmp_df, "Model", "Score", "Precison Comparison")
fig.write_image("public\\img\\precision-graph.jpeg")

# Recall Graph
clf_report_arr = [(name, model_dict[name].clf_report) for name in model_dict]
tmp_df = get_df_type(clf_report_arr, "Recall")
fig = pfr_graph(tmp_df, "Model", "Score", "Recall Comparison")
fig.write_image("public\\img\\recall-graph.jpeg")

# F1-Score Graph
clf_report_arr = [(name, model_dict[name].clf_report) for name in model_dict]
tmp_df = get_df_type(clf_report_arr, "F1-Score")
fig = pfr_graph(tmp_df, "Model", "Score", "F1-Score Comparison")
fig.write_image("public\\img\\f1-score-graph.jpeg")

# Accuracy Graph
acc_arr = [(key, model_dict[key].accuracy) for key in model_dict]
fig = acc_graph(acc_arr, "Accuracy Score Comparison", "Accuracy Score", "Types of Model")
fig.write_image("public\\img\\acc_score-graph.jpeg")

print("Complete all operation!")

# Output Res_DF
with open("Dataset\\model_result_dataFrame.json", 'w') as fout:
    json_dumps_str = json.dumps(res_df.to_dict(), indent=4)
    print(json_dumps_str, file=fout)
    
print("Successfully output model_result_dataFrame.json!")

# Output Correlation_Dataframe.json
with open("Dataset\\corr_dataFrame.json", 'w') as fout:
    json_dumps_str = json.dumps(cfs_dict, indent=4)
    print(json_dumps_str, file=fout)
    
print("Successfully output corr_dataFrame.json!")

sys.stdout.flush()