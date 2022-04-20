import sys
import numpy as np
import pandas as pd

import plotly.express as px
from Visualization.model_graph import *

selected_feature = sys.argv[1]
target_feature = sys.argv[2]

# Read Dataset
df = pd.read_csv("Dataset\\res_processed.csv")

# Construct Bar Graph that shows relationship between the two features
if selected_feature != "None->None":
    tmp_df = df.loc[:, [target_feature, selected_feature]]
    tmp_df.columns = [target_feature, "Target"]
    unique_arr = tmp_df[target_feature].unique().tolist()

    tmp_df = pd.crosstab(tmp_df.iloc[:, 1], tmp_df.iloc[:, 0])

    tmp_df = tmp_df[unique_arr]

    fig = show_bar_graph(tmp_df, selected_feature, target_feature, "Legend", "Count")
else:
    tmp_df = pd.DataFrame(df[target_feature].value_counts()).T
    fig = show_bar_graph(tmp_df, "", target_feature, "", "Count")

fig.write_image("public\\img\\filter-target-graph.jpeg")

print("Successfully output filter-target-graph.jpeg!")

sys.stdout.flush()